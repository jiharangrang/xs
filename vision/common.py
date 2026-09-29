"""데모의 저장 경로, 터미널 입력과 OpenCV 미리보기를 제공한다."""

from datetime import datetime
from pathlib import Path
import select
import sys

import cv2

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
RESULTS = ROOT / "results"


def save_image(image, directory, prefix):
    """원본 배열을 고유 이름의 PNG로 저장하고 실제 경로를 반환한다."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지 저장 실패: {path}")
    return path


def terminal_command():
    """미리보기를 멈추지 않고 터미널에서 완성된 한 줄만 읽는다."""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    line = sys.stdin.readline()
    return "q" if line == "" else line.strip().lower()


def preview(title, image):
    """영상 창을 갱신하고 키 입력이나 창 닫기를 반환한다."""
    cv2.imshow(title, image)
    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
        return ord("q")
    return key


def create_window(title):
    """원본 해상도를 유지하면서 크기를 조절할 수 있는 표시 창을 만든다."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)


def run(main):
    """사용자 중단과 실행 오류를 짧게 알리고 표시 창을 정리한다."""
    try:
        return main() or 0
    except KeyboardInterrupt:
        print("\n종료했습니다.")
        return 0
    except Exception as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
