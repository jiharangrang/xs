"""같은 빔 추정 알고리즘을 시뮬레이션 깊이와 저장된 실물 깊이에 적용한다.
CAD 정답 오차와 결과 영상을 저장하며 카메라·모터에는 연결하지 않는다.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from perception.beam import BeamDetectionError, BeamEstimate, estimate_beam
from perception.depth import depth_points, project_points, rectify_image
from planning.motion_path import load_path
from simulation.beam import beam_reference
from simulation.camera import JOINT_NAMES, reference_camera_info, render_camera
from simulation.model import load_model, set_path_time


def reference_errors(estimate: BeamEstimate, reference: dict) -> dict:
    r"""추정 평면과 중심선을 독립적인 CAD 정답과 비교한다.

    $$\theta=\cos^{-1}(|v^T\hat v|),\quad
    e_d=|d-\hat d|,\quad e_w=|w-\hat w|,\quad e_c=\|c-\hat c\|$$

    방향의 부호는 동일한 직선을 뜻하므로 제거하며 각도는 도, 거리는 mm로 출력한다.
    """
    result = {}
    for name in ("normal", "axis"):
        value = getattr(estimate, name)
        if value is not None:
            # 방향 부호를 제외한 두 단위 벡터의 내적: $$s=|v^T\hat v|$$
            cosine = abs(float(value @ np.asarray(reference[name])))
            # 방향 차이를 도 단위로 변환: $$\theta_{deg}=\cos^{-1}(s)180/\pi$$
            result[f"{name}_error_deg"] = float(np.rad2deg(np.arccos(np.clip(cosine, 0, 1))))
    # 평면 수직 거리 차이를 mm로 표시: $$e_{d,mm}=1000|d-\hat d|$$
    result["plane_distance_error_mm"] = abs(estimate.plane_offset_m - reference["plane_offset_m"]) * 1000
    if estimate.width_m is not None:
        # 폭 차이를 mm로 표시: $$e_{w,mm}=1000|w-\hat w|$$
        result["width_error_mm"] = abs(estimate.width_m - reference["width_m"]) * 1000
        # 중심선 대표점의 차이를 mm로 표시: $$e_{c,mm}=1000\|c-\hat c\|$$
        result["centerline_point_error_mm"] = float(np.linalg.norm(estimate.centerline_point_m - reference["centerline_point_m"]) * 1000)
    return result


def draw_estimate(depth: np.ndarray, profile: dict, estimate: BeamEstimate | None, title: str) -> np.ndarray:
    r"""깊이 영상에 검출한 평면·모서리·중심선과 세 공간 방향을 겹친다.

    $$p(s)=c+sv$$

    c는 중심선 대표점, v는 표시할 길이·폭·법선 방향이며 s는 표시 길이다.
    """
    import cv2

    gray = np.interp(np.nan_to_num(depth), [0.10, 0.70], [0, 255]).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~np.isfinite(depth)] = 0
    text = "plane not found"
    if estimate is not None:
        contours, _ = cv2.findContours(estimate.plane_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (150, 150, 150), 2)
        if estimate.edge_lines_m is not None:
            for edge in estimate.edge_lines_m:
                pixels = project_points(edge, profile)
                if np.all(np.isfinite(pixels)):
                    cv2.line(image, tuple(np.rint(pixels[0]).astype(int)), tuple(np.rint(pixels[1]).astype(int)), (0, 255, 255), 3)
        if estimate.centerline_point_m is not None:
            # 중심선의 표시 구간: $$p(s)=c+st,\quad s\in\{-0.25,0.25\}$$
            line = estimate.centerline_point_m + np.array([-0.25, 0.25])[:, None] * estimate.axis
            pixels = project_points(line, profile)
            if np.all(np.isfinite(pixels)):
                cv2.line(image, tuple(np.rint(pixels[0]).astype(int)), tuple(np.rint(pixels[1]).astype(int)), (0, 255, 0), 2)
            for label, direction, color in (("t", estimate.axis, (0, 255, 0)),
                                             ("b", estimate.width_axis, (0, 255, 255)),
                                             ("n", estimate.normal, (255, 255, 255))):
                # 축 화살표의 끝점: $$p_{end}=c+0.03v$$
                endpoint = estimate.centerline_point_m + 0.03 * direction
                ends = project_points(np.array([estimate.centerline_point_m, endpoint]), profile)
                if np.all(np.isfinite(ends)):
                    start, end = (tuple(np.rint(pixel).astype(int)) for pixel in ends)
                    cv2.arrowedLine(image, start, end, color, 3, tipLength=0.15)
                    cv2.putText(image, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        width = "unknown" if estimate.width_m is None else f"{estimate.width_m * 1000:.2f} mm"
        text = f"{estimate.status} / width {width} / plane RMS {estimate.plane_rms_m * 1000:.3f} mm"
    size = (640, round(image.shape[0] * 640 / image.shape[1]))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, 58, 0, 0, 0, cv2.BORDER_CONSTANT)
    for y, line in ((22, title), (46, text)):
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return image


def process_depth(depth: np.ndarray, profile: dict, folder: Path, name: str,
                  roi: tuple[int, int, int, int] | None = None) -> tuple[BeamEstimate | None, dict, np.ndarray]:
    """깊이를 추정 모듈에 전달하고 결과·평면 점군·표시 영상을 준비한다."""
    started = time.perf_counter()
    try:
        estimate = estimate_beam(depth, profile, roi=roi)
        result = estimate.as_dict()
    except BeamDetectionError as error:
        estimate, result = None, {"status": "not_found", "reason": str(error)}
    result["elapsed_s"] = time.perf_counter() - started
    rectified = rectify_image(depth, profile, depth=True)
    if estimate is not None:
        points = depth_points(rectified, profile)[estimate.plane_mask]
        np.save(folder / f"{name}_plane_points_m.npy", points[::max(1, len(points) // 20000)], allow_pickle=False)
    return estimate, result, draw_estimate(rectified, profile, estimate, name)


def main(argv: list[str] | None = None) -> int:
    """저장된 실물 깊이와 경로 자세의 가상 깊이에서 빔 정보를 추정한다."""
    parser = argparse.ArgumentParser(description="실물·시뮬레이션 빔 추정 시험")
    parser.add_argument("--capture", type=Path, help="test_camera 또는 compare_camera로 저장한 폴더")
    parser.add_argument("--path", type=Path, default=Path("outputs/rear_pull.json"), help="관절 기록이 없는 경우 시험할 시뮬레이션 경로")
    parser.add_argument("--time", type=float, default=0, help="시뮬레이션 경로의 재생 시각(초)")
    parser.add_argument("--roi", type=int, nargs=4, help="실물 빔 영역: 왼쪽 위 오른쪽 아래 픽셀")
    parser.add_argument("--output", type=Path, default=Path("outputs/beam"))
    args = parser.parse_args(argv)
    try:
        import cv2

        info, capture = reference_camera_info(), None
        if args.capture:
            metadata = args.capture / "comparison.json"
            if not metadata.exists():
                metadata = args.capture / "capture.json"
            capture = json.loads(metadata.read_text(encoding="utf-8"))
            info = capture["camera"]
        model, data = load_model()
        same_pose = bool(capture and capture.get("pose_source") == "calibration_server_measured")
        anchor = None
        if same_pose:
            angles = capture["joints_after"]["angles_deg"]
            for name, angle in angles.items():
                data.joint(name).qpos[0] = np.deg2rad(angle)
            mujoco.mj_forward(model, data)
        else:
            path = load_path(args.path)
            segment = set_path_time(model, data, path, args.time)
            anchor = path.segments[segment].anchor
            angles = {name: float(np.rad2deg(data.joint(name).qpos[0])) for name in JOINT_NAMES}
        reference = beam_reference(model, data)
        sim = render_camera(angles, camera_info=info, include_beam=True, anchor=anchor)
        folder = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder.mkdir(parents=True)
        sim_profile = {**info["depth"], "distortion": {}}
        estimate, sim_result, sim_image = process_depth(sim.depth_m, sim_profile, folder, "Simulation")
        if estimate is not None:
            sim_result["cad_errors"] = reference_errors(estimate, reference)
        record = {"cad_reference": reference, "simulation": sim_result, "camera": info,
                  "sim_joint_angles_deg": angles, "same_joint_pose": same_pose,
                  "real_roi": args.roi,
                  "real_to_sim_pose_error_evaluated": False,
                  "note": "실물 빔 배치는 CAD와 일치한다고 검증하지 않았으므로 자세 차이를 처짐으로 해석하지 않습니다."}
        images = [sim_image]
        np.save(folder / "sim_depth_m.npy", sim.depth_m, allow_pickle=False)
        if args.capture:
            name = "real_depth_m.npy" if (args.capture / "real_depth_m.npy").exists() else "depth_m.npy"
            real_depth = np.load(args.capture / name, allow_pickle=False)
            _, real_result, real_image = process_depth(real_depth, info["depth"], folder, "Real",
                                                       None if args.roi is None else tuple(args.roi))
            record["real"] = real_result
            record["capture_source"] = str(args.capture.resolve())
            images.append(real_image)
            if not same_pose:
                print("실물 촬영 당시 관절각 기록이 없어, 시뮬레이션은 별도 자세의 알고리즘 검증입니다.")
        canvas = np.hstack(images)
        if args.capture:
            caption = ("Recorded joint angles; beam placement not registered" if same_pose
                       else "Independent poses: detection comparison only, not sag measurement")
            canvas = cv2.copyMakeBorder(canvas, 0, 30, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(canvas, caption, (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        if not cv2.imwrite(str(folder / "comparison.png"), canvas):
            raise OSError("비교 영상 저장에 실패했습니다.")
        (folder / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ("simulation", "real"):
            if name in record:
                print(name, json.dumps(record[name], ensure_ascii=False, allow_nan=False))
        print(f"저장: {folder.resolve()}")
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"빔 추정 시험 실패: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
