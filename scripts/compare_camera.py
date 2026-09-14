"""정지한 실물의 RGB·깊이를 같은 관절각의 CAD 영상과 나란히 저장한다.
관절각은 실행 중인 캘리브레이션 서버에서 읽으며 모터 명령을 보내지 않는다.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from hardware.camera import CameraError, Gemini215Camera, RGBDFrame
from simulation.camera import JOINT_NAMES, render_camera
from perception.depth import rectify_image


def read_joint_snapshot(url: str) -> dict:
    """기존 서버의 새 상태 메시지에서 아홉 관절의 각도와 수신 시각을 얻는다."""
    from websockets.sync.client import connect

    with connect(url, open_timeout=5, close_timeout=1) as socket:
        payload = json.loads(socket.recv(timeout=5))
    states = payload["joints"]
    angles = {state["name"]: state["position_deg"] for state in states}
    if payload.get("error") or any(state.get("error") for state in states):
        raise ValueError("모든 모터의 상태를 읽지 못했습니다.")
    if set(angles) != set(JOINT_NAMES) or any(value is None for value in angles.values()):
        raise ValueError("아홉 관절의 실측 각도가 모두 필요합니다.")
    if not np.all(np.isfinite(list(angles.values()))):
        raise ValueError("관절각이 유한한 값이 아닙니다.")
    return {"angles_deg": angles, "states": states, "received_monotonic_s": time.monotonic()}


def depth_color(depth_m: np.ndarray) -> np.ndarray:
    """실제와 시뮬레이션에 같은 거리 색상을 적용하고 무효 깊이는 검게 표시한다."""
    import cv2

    gray = np.interp(np.nan_to_num(depth_m), [0.05, 0.70], [0, 255]).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~np.isfinite(depth_m)] = 0
    return image


def contour_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """실제 영상 위에 시뮬레이션 앞 그리퍼의 윤곽을 초록색으로 겹친다."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 3)
    return output


def panel(image: np.ndarray, label: str, *, depth: bool = False) -> np.ndarray:
    """영상의 가로세로 비율을 유지하면서 제목을 붙인 비교 패널을 만든다."""
    import cv2

    width = 640
    height = round(image.shape[0] * width / image.shape[1])
    scaled = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST if depth else cv2.INTER_AREA)
    labeled = cv2.copyMakeBorder(scaled, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(labeled, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)
    return labeled


def save_png(path: Path, image: np.ndarray) -> None:
    """PNG 저장 성공 여부를 확인한다."""
    import cv2

    if not cv2.imwrite(str(path), image):
        raise OSError(f"영상 저장 실패: {path}")


def main(argv: list[str] | None = None) -> int:
    """한 번 촬영하고 같은 관절각의 시뮬레이션과 원본·비교 영상을 저장한다."""
    parser = argparse.ArgumentParser(description="정지 자세의 실제 카메라와 CAD 비교")
    parser.add_argument("--server", default="ws://127.0.0.1:8000/ws", help="기존 캘리브레이션 서버")
    parser.add_argument("--serial", help="카메라 일련번호")
    parser.add_argument("--sim-only", action="store_true", help="실제 카메라 없이 시뮬레이션만 저장")
    parser.add_argument("--replay", type=Path, help="저장된 비교 폴더의 영상·관절각으로 다시 렌더링")
    parser.add_argument("--zero-pose", action="store_true", help="실측 대신 모든 관절을 0도로 가정")
    parser.add_argument("--include-beam", action="store_true", help="CAD에 저장된 빔 배치도 표시")
    parser.add_argument("--output", type=Path, default=Path("outputs/camera_comparison"))
    args = parser.parse_args(argv)
    if args.replay and (args.sim_only or args.zero_pose or args.include_beam):
        parser.error("--replay는 저장된 자세와 빔 표시 설정을 그대로 사용합니다.")
    try:
        import cv2

        zero = {"angles_deg": dict.fromkeys(JOINT_NAMES, 0.0)}
        frame, info = None, None
        if args.replay:
            saved = json.loads((args.replay / "comparison.json").read_text(encoding="utf-8"))
            bgr = cv2.imread(str(args.replay / "real_rgb.png"))
            if bgr is None:
                raise ValueError("저장된 실제 RGB 영상을 읽지 못했습니다.")
            frame = RGBDFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                              np.load(args.replay / "real_depth_m.npy", allow_pickle=False), **saved["frame"])
            before, after, info = saved["joints_before"], saved["joints_after"], saved["camera"]
            args.zero_pose = saved["pose_source"] == "assumed_zero"
            args.include_beam = saved["include_beam"]
        elif args.sim_only:
            before = after = zero if args.zero_pose else read_joint_snapshot(args.server)
        else:
            with Gemini215Camera(args.serial) as camera:
                print("카메라 준비: 30프레임 수신 후 정지 자세를 촬영합니다.", flush=True)
                for _ in range(30):
                    camera.read()
                before = zero if args.zero_pose else read_joint_snapshot(args.server)
                frame = camera.read()
                after = zero if args.zero_pose else read_joint_snapshot(args.server)
                info = camera.info.copy()
        print("비교 관절각(도):", after["angles_deg"], flush=True)
        sim = render_camera(after["angles_deg"], camera_info=info, include_beam=args.include_beam)
        folder = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder.mkdir(parents=True)
        sim_rgb = cv2.cvtColor(sim.rgb, cv2.COLOR_RGB2BGR)
        sim_depth = depth_color(sim.depth_m)
        save_png(folder / "sim_rgb.png", sim_rgb)
        save_png(folder / "sim_depth.png", sim_depth)
        np.save(folder / "sim_depth_m.npy", sim.depth_m, allow_pickle=False)
        record = {"pose_source": "assumed_zero" if args.zero_pose else "calibration_server_measured",
                  "joints_before": before, "joints_after": after,
                  "include_beam": args.include_beam, "camera": sim.camera_info,
                  "depth_unit": "m", "depth_color_range_m": [0.05, 0.70],
                  "note": "정지 자세의 시각 비교이며 자동 장착 보정이나 처짐 추정 결과가 아닙니다."}
        if args.replay:
            record["replayed_from"] = str(args.replay.resolve())
        if frame is None:
            rgb_panel = panel(sim_rgb, "Simulation RGB")
            depth_panel = panel(sim_depth, "Simulation depth: 50-700 mm", depth=True)
            height = max(rgb_panel.shape[0], depth_panel.shape[0])
            canvas = np.hstack(tuple(cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0,
                                                        cv2.BORDER_CONSTANT) for image in (rgb_panel, depth_panel)))
        else:
            real_rgb = rectify_image(frame.rgb, info["color"])
            real_rgb = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR)
            real_depth_m = rectify_image(frame.depth_m, info["depth"], depth=True)
            real_depth = depth_color(real_depth_m)
            rgb_overlay = contour_overlay(real_rgb, sim.rgb_gripper_mask)
            depth_overlay = contour_overlay(real_depth, sim.depth_gripper_mask)
            canvas = np.vstack((
                np.hstack((panel(real_rgb, "Real RGB (undistorted)"), panel(sim_rgb, "Simulation RGB"),
                           panel(rgb_overlay, "Green: simulated front gripper"))),
                np.hstack((panel(real_depth, "Real depth: 50-700 mm", depth=True),
                           panel(sim_depth, "Simulation depth: 50-700 mm", depth=True),
                           panel(depth_overlay, "Green: simulated front gripper", depth=True))),
            ))
            for filename, image in (("real_rgb.png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)),
                                    ("real_rgb_rectified.png", real_rgb), ("rgb_overlay.png", rgb_overlay),
                                    ("real_depth.png", real_depth), ("depth_overlay.png", depth_overlay)):
                save_png(folder / filename, image)
            np.save(folder / "real_depth_m.npy", frame.depth_m, allow_pickle=False)
            np.save(folder / "real_depth_rectified_m.npy", real_depth_m, allow_pickle=False)
            valid = sim.depth_gripper_mask & np.isfinite(real_depth_m) & np.isfinite(sim.depth_m)
            record["frame"] = {"color_timestamp_us": frame.color_timestamp_us,
                               "depth_timestamp_us": frame.depth_timestamp_us,
                               "received_monotonic_s": frame.received_monotonic_s}
            record["gripper_depth_overlap_pixels"] = int(valid.sum())
            record["gripper_depth_valid_fraction"] = float(valid.sum() / max(1, sim.depth_gripper_mask.sum()))
            print(f"예상 그리퍼 영역의 실제 유효 깊이: {record['gripper_depth_valid_fraction']:.1%}")
        save_png(folder / "comparison.png", canvas)
        (folder / "comparison.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
        print(f"저장: {folder.resolve()}")
        return 0
    except (CameraError, ImportError, OSError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"카메라 비교 실패: {error}", file=sys.stderr)
        if "uvc_open" in str(error) and "-3" in str(error):
            print("프로젝트 터미널에서 sudo .venv/bin/python -m scripts.compare_camera 로 실행하세요.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
