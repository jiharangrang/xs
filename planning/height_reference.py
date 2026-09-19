"""도착으로 확인한 실측 평면과 그리퍼 기준점으로 재사용 가능한 높이 기준을 저장한다.
카메라의 기울기가 달라도 같은 그리퍼 지점의 높이를 비교하도록 거리 기준을 제공한다.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from planning.beam_alignment import unit_normal


DEFAULT_DEPTH_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "models/xs/stage5_depth_reference.json"


@dataclass(frozen=True)
class HeightReference:
    """도착 자세에서 측정한 빔 거리·법선과 깊이 좌표의 그리퍼 기준점을 담는다."""

    depth_m: float
    normal_camera: tuple[float, float, float]
    tip_camera_m: tuple[float, float, float]

    def __post_init__(self):
        """도착 기준의 거리·법선·좌표가 유효한지 확인하고 불변 자료형으로 보관한다."""
        if isinstance(self.depth_m, bool) or not isinstance(self.depth_m, (float, int)) or not np.isfinite(self.depth_m) or self.depth_m <= 0:
            raise ValueError("도착 기준 깊이는 미터 단위의 유한한 양수여야 합니다.")
        normal = unit_normal(self.normal_camera)
        point = np.asarray(self.tip_camera_m, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("도착 기준의 그리퍼 좌표가 유효하지 않습니다.")
        object.__setattr__(self, "normal_camera", tuple(normal.tolist()))
        object.__setattr__(self, "tip_camera_m", tuple(point.tolist()))

    @property
    def signed_height_m(self):
        r"""확인된 도착 자세에서 그리퍼 기준점과 빔 평면의 부호 있는 거리를 반환한다.

        $$g_*=d_*+n_*^Tr_*$$
        """
        # 실측 도착 자세의 동일 지점 높이: $$g_*=d_*+n_*^Tr_*$$
        return float(self.depth_m + np.dot(self.normal_camera, self.tip_camera_m))

    def camera_goal(self, normal, tip_camera_m):
        r"""같은 그리퍼 높이를 만드는 현재 자세의 카메라 목표 거리를 계산한다.

        $$d_{goal}=g_*-n^Tr$$
        """
        normal = unit_normal(normal)
        point = np.asarray(tip_camera_m, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("현재 그리퍼 기준점 좌표가 유효하지 않습니다.")
        # 현재 기울기의 영향을 제거한 카메라 목표 거리: $$d_{goal}=g_*-n^Tr$$
        return float(self.signed_height_m - normal @ point)


def load_height_reference(path=DEFAULT_DEPTH_REFERENCE_PATH):
    """원본 실측값과 그리퍼 기준점이 모두 저장된 높이 보정을 읽는다."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or document.get("schema_version") != 2
            or document.get("metric") != "camera_to_beam_plane_perpendicular_m"
            or document.get("height_reference") != "tip_R"
            or document.get("reference_frame") != "depth_frame"):
        raise ValueError("5단계 높이 기준에 실측 평면과 그리퍼 기준점이 필요합니다.")
    try:
        return HeightReference(document["target_depth_m"], document["reference_normal_camera"],
                               document["reference_tip_camera_m"])
    except (KeyError, TypeError) as error:
        raise ValueError("저장한 높이 기준의 거리·법선·그리퍼 좌표를 확인해 주세요.") from error
