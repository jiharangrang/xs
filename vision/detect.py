"""저장한 RGB 사진에서 빨간 영역을 찾아 박스 영상과 좌표를 저장한다."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from common import CAPTURES, RESULTS, create_window, preview, run, save_image
from detector import add_detector_arguments, annotate, detector_from_args


def main():
    """지정 사진 또는 가장 최근 촬영 사진을 검출하고 결과를 표시한다."""
    parser = argparse.ArgumentParser(description="사진의 빨간 스티커 검출; 사진 경로 생략 시 최신 촬영본 사용")
    parser.add_argument("image", nargs="?", type=Path, help="검출할 사진 경로")
    parser.add_argument("--no-preview", action="store_true", help="화면 없이 결과만 저장")
    add_detector_arguments(parser)
    args = parser.parse_args()
    path = args.image
    if path is None:
        photos = sorted(CAPTURES.glob("rgb_*.png"))
        if not photos:
            raise FileNotFoundError("촬영한 사진이 없습니다. 먼저 capture.py를 실행하고 Enter로 저장하세요.")
        path = photos[-1]
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    detector = detector_from_args(args)
    detections = detector.detect(frame)
    output = annotate(frame, detections)
    saved = save_image(output, RESULTS, path.stem)
    metadata = {"source": str(path.resolve()), "method": "opencv_hsv",
                "min_area": args.min_area, "min_width": args.min_width, "min_saturation": args.min_saturation,
                "min_value": args.min_value, "detections": [asdict(item) for item in detections]}
    saved.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"검출: {len(detections)}개\n박스 영상: {saved}\n좌표: {saved.with_suffix('.json')}", flush=True)
    if not detections:
        print("검출되지 않았습니다. 작은 스티커는 --min-area와 --min-width, 어두운 사진은 --min-value를 낮춰 보세요.", flush=True)
    if not args.no_preview:
        title = "Red sticker photo | Q / Esc: quit"
        create_window(title)
        while preview(title, output) not in (ord("q"), 27, 10, 13):
            pass


if __name__ == "__main__":
    raise SystemExit(run(main))
