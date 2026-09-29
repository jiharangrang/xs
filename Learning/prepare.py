"""CAD 메시를 볼록 조각으로 분해하고 6단계 경로와 고정 학습 장면을 준비한다.
원본 모델의 이상적인 고정 파지를 사용하며 물리 장치에는 연결하지 않는다.
"""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "4")

import coacd
import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull

from common import ARTIFACTS, GEOMS, JOINTS, MODEL, REPO, file_hash, signature, write_json
from trace import REFERENCE, ideal_trace
from support import original_beam_pose, check_ideal_path


def source_mesh(name):
    r"""원본 STL을 미터 단위로 읽고 해당 몸체 좌표로 변환한다.

    $$v_B=R_{BG}(s\odot v_{CAD})+p_{BG}$$

    s는 STL의 길이 배율이고 G는 XML의 메시 배치 좌표다.
    """
    xml = ET.parse(MODEL).getroot()
    geom = xml.find(f".//geom[@name='{name}']")
    asset = xml.find(f"./asset/mesh[@name='{geom.attrib['mesh']}']")
    path = MODEL.parent / asset.attrib["file"]
    mesh = trimesh.load_mesh(path)
    mesh.apply_scale(np.fromstring(asset.attrib.get("scale", "1 1 1"), sep=" "))
    repaired = False
    if name == "ibeam_mesh":
        z0, z1 = mesh.bounds[:, 2]
        cap_mask = np.all(np.isclose(mesh.triangles[:, :, 2], z0, atol=1e-7), axis=1)
        cap = mesh.submesh([np.flatnonzero(cap_mask)], append=True)
        cap.update_faces(cap.unique_faces())
        cap.remove_unreferenced_vertices()
        original_volume = abs(mesh.volume)
        mesh = trimesh.creation.extrude_triangulation(cap.vertices[:, :2], cap.faces, z1 - z0)
        mesh.apply_translation([0, 0, z0])
        if not np.isclose(mesh.volume, original_volume, rtol=1e-5):
            raise ValueError("빔 끝단 단면의 압출과 원본 부피가 다릅니다.")
        repaired = True
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError(f"닫힌 유효한 입체 메시가 아닙니다: {path}")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.fromstring(geom.attrib.get("quat", "1 0 0 0"), sep=" "))
    position = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ")
    # CAD 메시를 몸체 좌표로 옮긴 꼭짓점: $$v_B=R_{BG}v_G+p_{BG}$$
    mesh.vertices = mesh.vertices @ rotation.reshape(3, 3).T + position
    return mesh, {"path": str(path), "sha256": file_hash(path), "watertight": True,
                  "beam_endcap_extrusion": repaired, "vertices": len(mesh.vertices), "faces": len(mesh.faces)}


def extruded_parts(mesh):
    """일정 단면의 빔을 단면 삼각형의 볼록 합병과 압출로 정확하게 분할한다."""
    axis = int(np.argmax(mesh.extents))
    transverse = [i for i in range(3) if i != axis]
    low, high = mesh.bounds[:, axis]
    mask = np.all(np.isclose(mesh.triangles[:, :, axis], low, atol=1e-7), axis=1)
    cap = mesh.submesh([np.flatnonzero(mask)], append=True)
    points = cap.vertices[:, transverse]
    groups = [set(map(int, face)) for face in cap.faces]
    while True:
        best = None
        for i, first in enumerate(groups):
            first_area = ConvexHull(points[list(first)]).volume
            for j in range(i + 1, len(groups)):
                second = groups[j]
                if len(first & second) < 2:
                    continue
                second_area = ConvexHull(points[list(second)]).volume
                union = first | second
                union_area = ConvexHull(points[list(union)]).volume
                if abs(union_area - first_area - second_area) < 1e-11:
                    if best is None or union_area > best[0]:
                        best = (union_area, i, j, union)
        if best is None:
            break
        _, i, j, union = best
        groups[i] = union
        groups.pop(j)
    parts = []
    for group in groups:
        lower = cap.vertices[sorted(group)].copy()
        upper = lower.copy()
        lower[:, axis] = low
        upper[:, axis] = high
        hull = trimesh.convex.convex_hull(np.vstack((lower, upper)))
        parts.append((np.asarray(hull.vertices), np.asarray(hull.faces)))
    if not np.isclose(sum(trimesh.Trimesh(v, f).volume for v, f in parts), mesh.volume, rtol=1e-5):
        raise ValueError("빔 단면의 볼록 분할이 원본 부피를 보존하지 않습니다.")
    return parts


def geometry_cache(directory, threshold):
    """이전 배치의 유효성과 무관하게 원본과 분해 설정이 같은 기하 캐시만 검증한다."""
    directory = Path(directory)
    config = json.loads((directory / "scene.json").read_text())
    if signature({k: v for k, v in config.items() if k != "scene_id"}) != config["scene_id"]:
        raise ValueError("기하 캐시의 설정 식별자가 다릅니다.")
    if config["decomposition_threshold_m"] != threshold:
        raise ValueError("재사용할 그리퍼의 분해 설정이 다릅니다.")
    for path, digest in ((MODEL, config["model_sha256"]),
                         (MODEL.parent / "camera_frames.xml", config["camera_frames_sha256"]),
                         (directory / "geometry.npz", config["geometry_sha256"])):
        if file_hash(path) != digest:
            raise ValueError(f"기하 캐시 원본이 변경되었습니다: {path}")
    for info in config["parts"].values():
        if file_hash(info["path"]) != info["sha256"]:
            raise ValueError("기하 캐시의 원본 메시가 변경되었습니다.")
    return config, np.load(directory / "geometry.npz", allow_pickle=False)


def prepare(output, *, threshold=.0005, reference=REFERENCE, reuse_grippers=None):
    """기하 근사 설정과 원본 식별자를 기록하며 재사용 가능한 학습 장면을 저장한다."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"기존 결과는 덮어쓰지 않습니다: {output}")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("분해 허용값은 양수여야 합니다.")
    q, times, stages, trace_info = ideal_trace(reference)
    output.mkdir(parents=True)
    np.savez_compressed(output / "trace.npz", q=q, time_s=times, stage=stages)
    write_json(output / "trace.json", trace_info)
    config = {"format": "xs.learning.scene.v2", "scope": "moving_right_gripper_vs_beam",
              "excluded": ["fixed_left_gripper_support", "other_links", "self_collision"],
              "distance_definition": "minimum signed convex-part-pair distance; not whole-solid penetration depth",
              "decomposition_threshold_m": threshold, "seed": 7,
              "beam_pose": original_beam_pose(), "fixed_support": trace_info["support"],
              "trajectory_kind": "ideal_cad_kinematic_stages", "joint_names": list(JOINTS),
              "source_model": str(MODEL), "model_sha256": file_hash(MODEL),
              "camera_frames_sha256": file_hash(MODEL.parent / "camera_frames.xml"), "parts": {}}
    calibration_path = MODEL.parent / "calibration.yaml"
    document = yaml.safe_load(calibration_path.read_text())
    config["limits"] = [[document["joints"][name]["lower_rad"], document["joints"][name]["upper_rad"]] for name in JOINTS]
    config["calibration_sha256"] = file_hash(calibration_path)
    arrays = {}
    cached = None
    if reuse_grippers is not None:
        previous, cached = geometry_cache(reuse_grippers, threshold)
    coacd.set_log_level("warn")
    for name in GEOMS:
        mesh, info = source_mesh(name)
        print(f"메시 분해: {name} / {len(mesh.faces)}개 삼각형", flush=True)
        if name == "ibeam_mesh":
            parts = extruded_parts(mesh)
            info["decomposition"] = "exact_convex_cross_section_extrusion"
        elif cached is not None:
            count = previous["parts"][name]["count"]
            parts = [(cached[f"{name}_{i}_v"], cached[f"{name}_{i}_f"]) for i in range(count)]
            info["decomposition"] = "coacd"
        else:
            parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
                                    threshold=threshold, real_metric=True, preprocess_mode="off",
                                    resolution=2000, mcts_iterations=100, seed=7)
            info["decomposition"] = "coacd"
        if not parts:
            raise RuntimeError(f"메시 분해 실패: {name}")
        for index, (vertices, faces) in enumerate(parts):
            arrays[f"{name}_{index}_v"] = vertices
            arrays[f"{name}_{index}_f"] = faces
        config["parts"][name] = {**info, "count": len(parts), "original_volume_m3": mesh.volume,
                                  "parts_volume_m3": sum(trimesh.Trimesh(v, f).volume for v, f in parts)}
        print(f"분해 완료: {name} / {len(parts)}개 볼록 메시", flush=True)
    np.savez_compressed(output / "geometry.npz", **arrays)
    if cached is not None:
        cached.close()
    config["geometry_sha256"] = file_hash(output / "geometry.npz")
    config["scene_id"] = signature(config)
    write_json(output / "scene.json", config)
    from scene import DistanceScene
    dense, distances, check = check_ideal_path(DistanceScene(output), q)
    np.savez_compressed(output / "path_check.npz", q=dense, distance=distances)
    write_json(output / "path_check.json", check)
    print(f"장면 준비 완료: {output} / 경로 {len(q)}개 표본", flush=True)
    return config


def main():
    """원본 CAD와 이론 관측 자세에서 전처리만 실행한다."""
    parser = argparse.ArgumentParser(description="1–6단계 오프라인 학습 장면 준비")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "scene")
    parser.add_argument("--threshold", type=float, default=.0005, help="CoACD 근사 허용값(m), 정확도 보증값은 아님")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--reuse-grippers", type=Path, help="원본·설정이 일치하는 이전 장면의 그리퍼 분해만 재사용")
    args = parser.parse_args()
    prepare(args.output, threshold=args.threshold, reference=args.reference,
            reuse_grippers=args.reuse_grippers)


if __name__ == "__main__":
    main()
