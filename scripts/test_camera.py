"""카메라 입력만 시험하고 RGB·깊이·장치 정보를 실행별 폴더에 저장한다.
선택적으로 RGB와 깊이를 나란히 표시하며 모터에는 연결하지 않는다.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from hardware.camera import CameraError, Gemini215Camera


def _preview(rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """RGB와 고정 거리 범위의 깊이 색상 영상을 나란히 만든다."""
    import cv2

    valid = np.isfinite(depth_m)
    gray = np.interp(np.nan_to_num(depth_m), [0.15, 0.70], [0, 255]).astype(np.uint8)
    depth_color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    color_panel = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 400))
    depth_panel = cv2.resize(depth_color, (640, 400), interpolation=cv2.INTER_NEAREST)
    cv2.putText(color_panel, "RGB", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_panel, "Depth 0.15-0.70 m / black: invalid", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.hstack((color_panel, depth_panel))


def main(argv: list[str] | None = None) -> int:
    """정해진 수의 프레임을 읽어 수신 상태와 마지막 원본 영상을 저장한다."""
    parser = argparse.ArgumentParser(description="Gemini 215 RGB·깊이 입력 테스트")
    parser.add_argument("--serial", help="카메라 일련번호, 한 개만 연결했다면 생략")
    parser.add_argument("--frames", type=int, default=60, help="수신할 프레임 수")
    parser.add_argument("--fps", type=int, default=15, help="RGB·깊이 공통 프레임 속도")
    parser.add_argument("--preview", action="store_true", help="실시간 화면 표시, Q 또는 Esc로 종료")
    parser.add_argument("--output", type=Path, default=Path("outputs/camera"), help="실행별 저장 폴더의 상위 경로")
    args = parser.parse_args(argv)
    if args.frames < 2 or args.fps <= 0:
        parser.error("--frames는 2 이상, --fps는 양수여야 합니다.")
    try:
        import cv2

        samples = []
        with Gemini215Camera(args.serial, fps=args.fps) as camera:
            print(f"연결: {camera.info['name']} / {camera.info['serial']} / {camera.info['connection']}", flush=True)
            print("RGB와 깊이는 정합 전 원본입니다. 같은 픽셀을 같은 위치로 해석하지 마세요.", flush=True)
            for index in range(args.frames):
                frame = camera.read()
                valid = np.isfinite(frame.depth_m)
                samples.append({
                    "color_timestamp_us": frame.color_timestamp_us,
                    "depth_timestamp_us": frame.depth_timestamp_us,
                    "received_monotonic_s": frame.received_monotonic_s,
                    "valid_depth_fraction": float(valid.mean()),
                    "median_depth_m": float(np.median(frame.depth_m[valid])) if valid.any() else None,
                })
                if index == 0 or (index + 1) % 15 == 0:
                    print(f"수신 {index + 1}/{args.frames}: RGB {frame.rgb.shape}, 깊이 {frame.depth_m.shape}, "
                          f"유효 깊이 {valid.mean():.1%}", flush=True)
                if args.preview:
                    cv2.imshow("Gemini 215 - RGB / native depth", _preview(frame.rgb, frame.depth_m))
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            info = camera.info.copy()
        folder = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder.mkdir(parents=True)
        for name, image in (("rgb.png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)),
                            ("preview.png", _preview(frame.rgb, frame.depth_m))):
            if not cv2.imwrite(str(folder / name), image):
                raise CameraError(f"영상을 저장하지 못했습니다: {folder / name}")
        np.save(folder / "depth_m.npy", frame.depth_m, allow_pickle=False)
        elapsed = samples[-1]["received_monotonic_s"] - samples[0]["received_monotonic_s"]
        summary = {"camera": info, "frames_received": len(samples),
                   "received_fps": (len(samples) - 1) / elapsed if elapsed > 0 else None,
                   "samples": samples}
        (folder / "capture.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"저장: {folder.resolve()}")
        print(f"수신 완료: {len(samples)} 프레임")
        if not any(sample["valid_depth_fraction"] > 0 for sample in samples):
            raise CameraError("영상은 수신했지만 유효 깊이가 없습니다. 대상 거리·시야를 확인하세요.")
        return 0
    except (CameraError, ImportError, OSError, RuntimeError) as error:
        print(f"카메라 테스트 실패: {error}", file=sys.stderr)
        if sys.platform == "darwin" and "uvc_open" in str(error) and "-3" in str(error):
            print("macOS USB 접근 권한 오류입니다. 의존성 설치 후 터미널에서 "
                  "sudo .venv/bin/python -m scripts.test_camera 로 다시 실행하세요.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if args.preview and "cv2" in locals():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
