"""영상과 같은 경로의 관절 제한·고정단·도착 위치·메시 충돌을 표본 검사한다.
표본 사이의 연속 충돌 증명이나 접촉력·동역학 검증으로 해석하지 않는다.
"""

import hashlib
import json
import time

import numpy as np

from kinematics.anchoring import TipAnchor
from kinematics.joint_limits import load_joint_limits
from kinematics.joints import ARM_JOINT_NAMES
from Corner.scene import OUTPUT, SOURCE, Scene, site_pose
from Corner.trajectory import Trajectory


def digest(path):
    """파일 내용의 재현성 식별자를 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_anchor_switch(scene, arrays):
    """고정단 교대 때 양쪽 턱이 닫혔고 로봇의 모든 메시 위치가 연속인지 확인한다."""
    maximum = 0.
    switches = 0
    for index in range(1, len(arrays["q"])):
        if arrays["fixed_tip"][index] == arrays["fixed_tip"][index - 1]:
            continue
        switches += 1
        if not np.allclose(arrays["grippers"][index - 1:index + 1], 0, atol=1e-10):
            raise ValueError("양쪽 그리퍼가 닫히기 전에 고정단이 바뀌었습니다.")
        if not np.allclose(arrays["q"][index - 1], arrays["q"][index], atol=1e-10):
            raise ValueError("고정단 교대 구간에 관절 이동이 있습니다.")
        positions = []
        for j in (index - 1, index):
            scene.set(arrays["q"][j], arrays["grippers"][j],
                      TipAnchor(str(arrays["fixed_tip"][j]), arrays["anchor"][j]))
            positions.append(scene.data.geom_xpos[scene.robot].copy())
        maximum = max(maximum, float(np.max(np.abs(positions[0] - positions[1]))))
    if switches != 1 or maximum > 1e-7:
        raise ValueError("고정단 교대 횟수 또는 위치 연속성 검사가 실패했습니다.")
    return maximum


def validate(directory=OUTPUT, step_deg=.25):
    r"""관절 보간 구간을 잘게 나누어 실제 CAD 메시 교차를 검사하고 결과를 저장한다.

    $$N_i=\max(1,\lceil\|q_{i+1}-q_i\|_\infty/h\rceil)$$

    모든 로봇 메시와 두 빔을 검사하고 자기충돌은 동일·직접 연결 몸체를 제외한다.
    """
    if not np.isfinite(step_deg) or not 0 < step_deg <= 1:
        raise ValueError("검사 관절 간격은 0도 초과 1도 이하여야 합니다.")
    trajectory = Trajectory(directory)
    arrays = trajectory.arrays
    scene = Scene(directory / "scene/model.xml")
    names = (*ARM_JOINT_NAMES, "G_L", "G_R")
    limits = np.asarray(list(load_joint_limits(names).values()))
    angles = np.column_stack([arrays["q"], arrays["grippers"]])
    if np.any(angles < limits[:, 0] - 1e-8) or np.any(angles > limits[:, 1] + 1e-8):
        raise ValueError("저장 경로가 기존 관절 제한을 벗어납니다.")
    switch_error = check_anchor_switch(scene, arrays)
    for index, side in enumerate(arrays["fixed_tip"]):
        grip_index = 0 if side == "tip_L" else 1
        if abs(arrays["grippers"][index, grip_index]) > 1e-8:
            raise ValueError("고정 팁의 그리퍼가 열려 있습니다.")
    q, grippers, anchor, _ = trajectory.sample(0)
    collision_anchor = anchor.T_world_fixed_tip.copy()
    collision_anchor[2, 3] -= .004
    scene.set(q, grippers, TipAnchor(anchor.fixed_tip, collision_anchor))
    if not any(record[3] for record in scene.beam_distances()):
        raise ValueError("의도적으로 겹친 대조 자세에서 충돌 검출기가 반응하지 않았습니다.")
    count, minimum, anchor_error = 0, 1., 0.
    collisions = []
    started = time.monotonic()
    for index in range(len(angles) - 1):
        # 한 구간에서 가장 많이 변한 관절각: $$\delta_i=\|q_{i+1}-q_i\|_\infty$$
        change = np.max(np.abs(angles[index + 1] - angles[index]))
        # 최대 관절 간격에 맞는 세부 표본 수: $$N_i=\max(1,\lceil\delta_i/h\rceil)$$
        divisions = max(1, int(np.ceil(change / np.deg2rad(step_deg))))
        fractions = np.linspace(0, 1, divisions + 1)
        for fraction in fractions if index == 0 else fractions[1:]:
            q, grippers, anchor, phase = trajectory.segment(index, fraction)
            scene.set(q, grippers, anchor)
            actual = site_pose(scene.data, anchor.fixed_tip)
            anchor_error = max(anchor_error, float(np.max(np.abs(actual - anchor.T_world_fixed_tip))))
            records = scene.beam_distances()
            minimum = min(minimum, min(record[2] for record in records))
            beam_hits = [record[:2] for record in records if record[3]]
            self_hits = scene.self_collisions()
            if beam_hits or self_hits:
                collisions.append({"segment": index, "fraction": float(fraction), "phase": phase,
                                   "beam": beam_hits, "self": self_hits})
            count += 1
        if index % 60 == 0:
            print(f"검사: 구간 {index}/{len(angles) - 1}, 표본 {count}, 충돌 {len(collisions)}", flush=True)
    settings = trajectory.meta["settings"]
    final_error = 0.
    for side, y in (("L", settings["rear_y"]), ("R", settings["front_y"])):
        target = [settings["corner_x"], y, settings["grasp_z"]]
        error = np.linalg.norm(scene.data.site(f"tip_{side}").xpos - target)
        final_error = max(final_error, float(error))
    if not np.allclose(grippers, 0, atol=1e-10):
        raise ValueError("마지막에 양쪽 턱이 닫히지 않았습니다.")
    passed = not collisions and anchor_error < 1e-7 and final_error < 1e-4
    result = {"passed": passed, "scope": "sampled_kinematic_CAD_mesh_intersection",
              "samples": count, "max_joint_sample_step_deg": step_deg,
              "beam_collision_samples": sum(bool(c["beam"]) for c in collisions),
              "self_collision_samples": sum(bool(c["self"]) for c in collisions),
              "minimum_robot_beam_distance_m": minimum, "anchor_matrix_error_max": anchor_error,
              "anchor_switch_geom_position_error_m": switch_error, "final_tip_position_error_m": final_error,
              "negative_control_detected": True, "duration_s": trajectory.duration,
              "check_seconds": time.monotonic() - started,
              "excluded_self_pairs": "same body and directly connected parent-child bodies",
              "limitations": ["sampled check, not continuous collision certification",
                              "triangle surface intersections, not a volumetric containment test",
                              "ideal fixed gripper, no friction, load, compliance or motor dynamics",
                              "nominal grasp has a small CAD clearance; no contact force is simulated"],
              "trajectory_sha256": digest(directory / "trajectory.npz"),
              "scene_sha256": digest(directory / "scene/model.xml"),
              "source_sha256": {str(p.relative_to(SOURCE.parent)): digest(p)
                                for p in sorted(SOURCE.parent.rglob("*"))
                                if p.is_file() and p.suffix in (".xml", ".yaml", ".stl")},
              "collisions": collisions}
    (directory / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"경로 검사 실패: 충돌 {len(collisions)}개, 결과를 확인하세요.")
    print(f"검사 통과: {count}개 표본, 빔 충돌 0, 자기충돌 0", flush=True)
    return result
