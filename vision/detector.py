"""HSV 색상 범위로 빨간 영역을 찾고 작은 잡음을 제거해 박스로 표시한다.
학습 모델 없이 색상을 검출하며 스티커와 다른 빨간 물체를 구별하지 않는다.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Detection:
    """빨간 영역의 픽셀 좌표 박스와 윤곽 면적을 보관한다."""

    xyxy: tuple[int, int, int, int]
    area_px: float


class StickerDetector:
    """색상·최소 면적·최소 폭 조건으로 빨간 스티커 후보를 찾는다."""

    def __init__(self, *, min_area=2000, min_width=25, min_saturation=80, min_value=50):
        """원본 영상에서 사용할 최소 면적·폭·채도·밝기를 지정한다."""
        if not math.isfinite(min_area) or min_area <= 0:
            raise ValueError("min-area는 양의 유한한 값이어야 합니다.")
        if not math.isfinite(min_width) or min_width < 0:
            raise ValueError("min-width는 0 이상의 유한한 값이어야 합니다.")
        if not 0 <= min_saturation <= 255 or not 0 <= min_value <= 255:
            raise ValueError("min-saturation과 min-value는 0~255여야 합니다.")
        self.min_area = min_area
        self.min_width = min_width
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.kernel = np.ones((3, 3), dtype=np.uint8)

    def mask(self, bgr):
        """색상환 양 끝의 빨간색을 합치고 작은 점과 내부 틈을 정리한다."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = cv2.inRange(hsv, (0, self.min_saturation, self.min_value), (10, 255, 255))
        high = cv2.inRange(hsv, (170, self.min_saturation, self.min_value), (179, 255, 255))
        mask = cv2.bitwise_or(low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

    def detect(self, bgr):
        """면적과 회전 박스의 짧은 변을 확인해 작거나 가느다란 빨간 영역을 거른다."""
        contours, _ = cv2.findContours(self.mask(bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if min(cv2.minAreaRect(contour)[1]) < self.min_width:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(Detection((x, y, x + width, y + height), area))
        return sorted(detections, key=lambda item: item.xyxy)


@lru_cache(maxsize=1)
def korean_font():
    """설치된 한글 글꼴을 한 번 읽어 사진과 실시간 표시에서 재사용한다."""
    for path in (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=72)
    raise RuntimeError("한글 표시용 글꼴이 없습니다. Noto Sans CJK 또는 나눔고딕을 설치하세요.")


def annotate(bgr, detections):
    """검출 박스를 그리고 좌상단에 큰 한글로 부식 개수를 표시한다."""
    output = bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
    canvas = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = korean_font()
    draw.text((24, 24), f"부식 : {len(detections)}개", font=font, fill=(255, 255, 0), anchor="lt", stroke_width=1)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def add_detector_arguments(parser):
    """사진과 실시간 검출에서 같은 색상·크기 설정을 사용하도록 한다."""
    parser.add_argument("--min-area", type=float, default=2000, help="원본에서 검출할 최소 윤곽 면적, 기본 2000")
    parser.add_argument("--min-width", type=float, default=25, help="회전 박스 짧은 변의 최소 길이(픽셀), 기본 25; 0이면 폭 검사 해제")
    parser.add_argument("--min-saturation", type=int, default=80, help="최소 채도 0~255, 기본 80")
    parser.add_argument("--min-value", type=int, default=50, help="최소 밝기 0~255, 기본 50")


def detector_from_args(args):
    """명령행 설정으로 사진과 실시간에서 동일한 검출기를 만든다."""
    return StickerDetector(min_area=args.min_area, min_width=args.min_width, min_saturation=args.min_saturation,
                           min_value=args.min_value)
