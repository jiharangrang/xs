"""정지 자세의 RGB·깊이 묶음과 실제 관절각, 촬영 구간의 모터 로그를 저장한다.
단독 카메라 또는 공유 수신기를 사용하며 모터 명령이나 빔 분석은 수행하지 않는다.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from hardware.camera import CameraError, Gemini215Camera


SCAN_FRAMES = 15
WARMUP_FRAMES = 30
DEFAULT_SCAN_DIR = Path(__file__).resolve().parents[1] / "outputs" / "camera_scans"


def capture_scan(read_pose: Callable[[], dict], motor_log_path: Path,
                 output_root: Path = DEFAULT_SCAN_DIR, *, camera_factory=None) -> dict:
    """카메라 안정화 후 15프레임을 수집하고 자세와 로그를 같은 폴더에 저장한다."""
    started_utc = datetime.now(timezone.utc)
    log_start = motor_log_path.stat().st_size
    factory = Gemini215Camera if camera_factory is None else camera_factory
    with factory() as camera:
        deadline = time.monotonic() + 20.0
        for _ in range(WARMUP_FRAMES):
            camera.read()
            if time.monotonic() > deadline:
                raise CameraError("카메라 준비 시간이 초과되었습니다. 다시 스캔해 주세요.")
        before = read_pose()
        moving = [state["name"] for state in before["states"] if state["speed_deg_s"] != 0]
        if moving:
            raise CameraError("아직 움직이는 관절이 있습니다: " + ", ".join(moving))
        frames = []
        for _ in range(SCAN_FRAMES):
            frames.append(camera.read())
            if time.monotonic() > deadline:
                raise CameraError("영상 수신 시간이 초과되었습니다. 다시 스캔해 주세요.")
        after = read_pose()
        info = camera.info.copy()
        log_end = motor_log_path.stat().st_size

    output = output_root / started_utc.strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    frame_dir = output / "frames"
    frame_dir.mkdir()
    import cv2

    samples = []
    for index, frame in enumerate(frames):
        rgb_file = f"frames/{index:03d}_rgb.png"
        depth_key = f"frame_{index:03d}"
        if not cv2.imwrite(str(output / rgb_file), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"RGB 영상을 저장하지 못했습니다: {rgb_file}")
        samples.append({"rgb_file": rgb_file, "depth_key": depth_key,
                        "color_timestamp_us": frame.color_timestamp_us,
                        "depth_timestamp_us": frame.depth_timestamp_us,
                        "received_monotonic_s": frame.received_monotonic_s,
                        "valid_depth_fraction": float(np.isfinite(frame.depth_m).mean())})
    np.savez_compressed(output / "depth_frames.npz",
                        **{sample["depth_key"]: frame.depth_m for sample, frame in zip(samples, frames)})
    np.save(output / "depth_m.npy", frames[-1].depth_m)
    (output / "rgb.png").write_bytes((output / samples[-1]["rgb_file"]).read_bytes())
    with motor_log_path.open("rb") as source, (output / "motors.jsonl").open("wb") as destination:
        source.seek(log_start)
        destination.write(source.read(log_end - log_start))
    record = {
        "schema_version": 1, "started_utc": started_utc.isoformat(),
        "camera": info, "frames_received": len(frames), "samples": samples,
        "depth_frames_file": "depth_frames.npz", "depth_unit": "m",
        "representative_frame": len(frames) - 1,
        "pose_source": "calibration_server_measured", "joints_before": before, "joints_after": after,
        "motor_log_file": "motors.jsonl", "motor_log_source": str(motor_log_path),
        "motor_log_byte_range": [log_start, log_end],
        "note": "RGB·깊이는 미정합 원본이며 대표 영상은 마지막 프레임이다. "
                "관절각은 촬영 전후에 순차 조회했다. 모터 설정·이동 이력은 원본 로그를 참조한다.",
    }
    (output / "capture.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "frames": len(frames),
            "valid_depth_fraction": samples[-1]["valid_depth_fraction"]}
