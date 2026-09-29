"""Gemini 215 RGB 영상에서 빨간 스티커를 검출해 실시간 박스로 표시한다."""

import argparse

from camera import RGBCamera
from common import RESULTS, create_window, preview, run, save_image, terminal_command
from detector import add_detector_arguments, annotate, detector_from_args


def main():
    """사진 검출과 같은 색상 조건으로 영상을 처리하고 검출 화면을 표시한다."""
    parser = argparse.ArgumentParser(description="실시간 빨간 스티커 검출, Enter로 결과 저장, Q 또는 Esc로 종료")
    parser.add_argument("--serial", help="카메라 일련번호")
    add_detector_arguments(parser)
    args = parser.parse_args()
    detector = detector_from_args(args)
    title = "Red sticker live | Enter: save | Q / Esc: quit"
    with RGBCamera(args.serial) as camera:
        create_window(title)
        print("실시간 검출 중: Enter로 박스 영상 저장, q + Enter 또는 영상 창의 Q/Esc로 종료", flush=True)
        while True:
            frame = camera.read()
            detections = detector.detect(frame)
            output = annotate(frame, detections)
            key = preview(title, output)
            command = terminal_command()
            if command == "q" or key in (ord("q"), 27):
                break
            if command == "" or key in (10, 13):
                print(f"저장: {save_image(output, RESULTS, 'live')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(run(main))
