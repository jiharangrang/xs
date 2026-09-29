"""RGB 화면을 보면서 터미널의 Enter 입력마다 원본 사진을 저장한다."""

import argparse

from camera import RGBCamera
from common import CAPTURES, create_window, preview, run, save_image, terminal_command


def main():
    """카메라 영상을 계속 갱신하면서 촬영 또는 종료 입력을 처리한다."""
    parser = argparse.ArgumentParser(description="Enter로 Gemini 215 RGB 사진 저장, q + Enter로 종료")
    parser.add_argument("--serial", help="카메라 일련번호")
    parser.add_argument("--no-preview", action="store_true", help="영상 창 없이 터미널만 사용")
    args = parser.parse_args()
    title = "RGB capture | Enter: save | Q / Esc: quit"
    with RGBCamera(args.serial) as camera:
        if not args.no_preview:
            create_window(title)
        print("터미널에서 Enter를 누르면 원본 PNG 저장, q + Enter로 종료합니다.", flush=True)
        print(f"저장 위치: {CAPTURES}", flush=True)
        while True:
            frame = camera.read()
            key = -1 if args.no_preview else preview(title, frame)
            command = terminal_command()
            if command == "q" or key in (ord("q"), 27):
                break
            if command == "" or key in (10, 13):
                print(f"저장: {save_image(frame, CAPTURES, 'rgb')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(run(main))
