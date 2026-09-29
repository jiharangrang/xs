"""초기 자세에서 20 cm 앞 삽입 완료까지 메시·SE3NN 경로 생성의 전체 시간을 비교한다.
기존 6단계 생성기를 매번 새로 실행하며 최종 메시 재검사 시간도 양쪽에 포함한다.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from common import ARTIFACTS, REPO, file_hash, write_json
from model import load_checkpoint
from path_planner import CollisionAwareIK, DistanceOracle, PathSettings, dense_path
from scene import DistanceScene
from support import require_initial_grasp

sys.path.insert(0, str(REPO))
from Learning.trace import REFERENCE, ideal_trace
from kinematics.poses import pose_error
from planning.targets import beam_grasp_target


def validate_path(scene, q, settings):
    r"""계획보다 조밀한 메시 검사로 간격·왼쪽 파지·20 cm 목표 도착을 확인한다.

    $${}^W T_{R,d}={}^W T_L{}^L T_{R,grasp},\qquad p_{d,z}\leftarrow p_{d,z}+0.001$$
    """
    dense = dense_path(q, settings.validation_subdivisions)
    distances, tip_points = [], []
    for sample in dense:
        distances.append(scene.distance(sample))
        require_initial_grasp(scene.model, scene.data)
        tip_points.append(scene.data.site("tip_R").xpos.copy())
    distances = np.asarray(distances)
    fixed = np.eye(4)
    fixed[:3, :3] = scene.data.site("tip_L").xmat.reshape(3, 3)
    fixed[:3, 3] = scene.data.site("tip_L").xpos
    # 기존 마주 보는 파지 자세를 월드 좌표로 표현: $${}^W T_{R,d}={}^W T_L{}^L T_{R,grasp}$$
    target = fixed @ beam_grasp_target(.2)
    # 기존 6단계와 같은 삽입 높이 여유: $$p_{d,z}\leftarrow p_{d,z}+0.001$$
    target[2, 3] += .001
    actual = np.eye(4)
    actual[:3, :3] = scene.data.site("tip_R").xmat.reshape(3, 3)
    actual[:3, 3] = scene.data.site("tip_R").xpos
    error = pose_error(actual, target)
    position_error = float(np.linalg.norm(error[:3]))
    rotation_error = float(np.linalg.norm(error[3:]))
    initial_valid = bool(np.allclose(q[0, :7], 0., atol=1e-8))
    jaw_valid = bool(np.allclose(q[:, -1], settings.jaw_rad, atol=1e-8))
    report = {"samples": len(dense), "minimum_distance_m": float(distances.min()),
              "penetrating_samples": int((distances < 0).sum()),
              "below_margin_samples": int((distances < settings.margin_m - 1e-7).sum()),
              "goal_position_error_m": position_error, "goal_rotation_error_rad": rotation_error,
              "goal_xyz_m": actual[:3, 3].tolist(), "target_xyz_m": target[:3, 3].tolist(),
              "initial_pose_valid": initial_valid, "jaw_open_all_samples": jaw_valid,
              "fixed_left_support_valid": True, "continuous_collision_proof": False,
              "tip_path_length_m": float(np.linalg.norm(np.diff(tip_points, axis=0), axis=1).sum()),
              "joint_path_length_rad": float(np.linalg.norm(np.diff(q[:, :7], axis=0), axis=1).sum())}
    report["success"] = bool(report["below_margin_samples"] == 0 and position_error < 1e-4
                             and rotation_error < 1e-3 and initial_valid and jaw_valid)
    return report, dense, distances


def build_engine(mode, scene_path, checkpoint):
    """첫 요청에 필요한 메시 장면과 선택적인 학습 가중치를 준비한다."""
    scene = DistanceScene(scene_path)
    model = None
    if mode == "learned":
        model, _ = load_checkpoint(checkpoint, scene)
    elif mode != "mesh":
        raise ValueError("거리 계산 방식은 mesh 또는 learned여야 합니다.")
    return DistanceOracle(scene, model)


def run_request(oracle, settings, reference):
    """기준 자세 IK·거리 제약 경로·최종 재검사를 포함한 한 요청을 실제 측정한다."""
    began = time.perf_counter()
    oracle.reset()
    solver = CollisionAwareIK(oracle, settings)
    q, time_s, stages, metadata = ideal_trace(reference, solver=solver, segment_guard=solver.guard_segment,
                                             recompute_reference=True)
    planning_finished = time.perf_counter()
    validation, dense, distances = validate_path(oracle.scene, q, settings)
    ended = time.perf_counter()
    report = {"planning_s": planning_finished - began, "mesh_validation_s": ended - planning_finished,
              "total_s": ended - began, "collision_query_s": oracle.elapsed_s,
              "collision_configurations": oracle.configurations, "collision_batches": oracle.calls,
              "ik_solve_calls": solver.solve_calls, "optimizer_iterations": solver.optimizer_iterations,
              "path_samples": len(q), "validation": validation, "path_metadata": metadata}
    arrays = {"q": q, "time_s": time_s, "stage": stages, "validation_q": dense, "true_distance": distances}
    return report, arrays


def summarize(entries):
    """동일 조건 반복 실행의 중앙값과 범위를 요약한다."""
    result = {"runs": len(entries), "success_count": sum(x["validation"]["success"] for x in entries)}
    for key in ("planning_s", "mesh_validation_s", "total_s", "collision_query_s", "collision_configurations",
                "collision_batches", "ik_solve_calls", "optimizer_iterations"):
        values = np.asarray([entry[key] for entry in entries])
        result[key] = {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}
    result["minimum_clearance_m"] = min(entry["validation"]["minimum_distance_m"] for entry in entries)
    result["maximum_goal_error_m"] = max(entry["validation"]["goal_position_error_m"] for entry in entries)
    return result


def benchmark(args):
    """같은 CPU에서 첫 요청과 교대 순서의 반복 요청을 구분해 비교한다."""
    if args.repeats < 1:
        raise ValueError("반복 횟수는 양수여야 합니다.")
    if args.output.exists():
        raise FileExistsError(f"기존 비교 결과를 덮어쓰지 않습니다: {args.output}")
    settings = PathSettings(margin_m=args.margin_mm / 1000)
    args.output.mkdir(parents=True)
    report = {"settings": asdict(settings), "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": file_hash(args.checkpoint), "reference_sha256": file_hash(args.reference),
              "execution": {"platform": platform.platform(), "torch_version": str(torch.__version__),
                            "torch_threads": torch.get_num_threads(), "device": "cpu"},
              "timing_scope": "기준·관측 자세 IK 재계산, 6단계 경로 생성, 거리 제약 및 8분할 메시 재검사",
              "first_request_scope": "장면·가중치 로딩과 첫 전체 요청; Python 시작·import 시간은 제외",
              "excluded": ["학습과 메시 분해 전처리", "저장 파일 쓰기", "카메라·모터·렌더링", "원래 기준 파지의 전역 IK 탐색"],
              "algorithm": "기존 6단계 직선 IK + SLSQP 구간 거리 제약; 전역 최단 경로 탐색은 아님",
              "distance_gradient": "두 방식 모두 0.0001 rad 유한차분; 같은 자세 묶음 입력",
              "learned_batching": "CPU 신경망은 후보 묶음을 배치 처리; 메시 정답은 MuJoCo 순차 질의",
              "cold": {}, "runs": {"mesh": [], "learned": []}}
    report["source_sha256"] = {name: file_hash(REPO / name) for name in (
        "Learning/path_planner.py", "Learning/path_benchmark.py", "Learning/trace.py", "Learning/scene.py",
        "Learning/model.py", "planning/linear_motion.py", "planning/targets.py", "kinematics/ik.py",
        "kinematics/anchoring.py")}
    engines = {}
    for mode in ("mesh", "learned"):
        began = time.perf_counter()
        engines[mode] = build_engine(mode, args.scene, args.checkpoint)
        setup_s = time.perf_counter() - began
        entry, arrays = run_request(engines[mode], settings, args.reference)
        entry["setup_s"] = setup_s
        entry["first_path_total_s"] = setup_s + entry["total_s"]
        report["cold"][mode] = entry
        report["scene_id"] = engines[mode].scene.scene_id
        report["collision_scope"] = engines[mode].scene.config["scope"]
        np.savez_compressed(args.output / f"{mode}_cold_path.npz", **arrays)
        write_json(args.output / "progress.json", report)
        print(f"첫 {mode}: {entry['first_path_total_s']:.3f} s, 검증 성공={entry['validation']['success']}", flush=True)
        if not entry["validation"]["success"]:
            raise RuntimeError(f"{mode} 경로가 최종 메시 검증에 실패했습니다. 결과에 성공으로 기록하지 않습니다.")
    representative = {}
    for repeat in range(args.repeats):
        order = ("mesh", "learned") if repeat % 2 == 0 else ("learned", "mesh")
        for mode in order:
            entry, arrays = run_request(engines[mode], settings, args.reference)
            entry["repeat"] = repeat
            report["runs"][mode].append(entry)
            if repeat == 0:
                representative[mode] = arrays
                np.savez_compressed(args.output / f"{mode}_path.npz", **arrays)
            write_json(args.output / "progress.json", report)
            print(f"반복 {repeat + 1}/{args.repeats} {mode}: 생성 {entry['planning_s']:.3f} s, "
                  f"재검사 {entry['mesh_validation_s']:.3f} s, 전체 {entry['total_s']:.3f} s", flush=True)
    report["summary"] = {mode: summarize(entries) for mode, entries in report["runs"].items()}
    mesh_q, learned_q = representative["mesh"]["q"], representative["learned"]["q"]
    report["same_path_shape"] = mesh_q.shape == learned_q.shape
    report["maximum_joint_path_difference_rad"] = float(np.max(np.abs(mesh_q - learned_q))) if mesh_q.shape == learned_q.shape else None
    report["total_speedup"] = report["summary"]["mesh"]["total_s"]["median"] / report["summary"]["learned"]["total_s"]["median"]
    report["all_paths_valid"] = all(entry["validation"]["success"] for entries in report["runs"].values() for entry in entries)
    write_json(args.output / "results.json", report)
    print(f"전체 요청 속도비: {report['total_speedup']:.3f}, 전체 경로 검증={report['all_paths_valid']}", flush=True)


def main():
    """학습 또는 모터 접속 없이 오프라인 경로 생성 시간을 비교한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=ARTIFACTS / "scene")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "mixed_1m_scratch/model.pt")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "path_planning_20cm")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--margin-mm", type=float, default=.5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    benchmark(args)


if __name__ == "__main__":
    main()
