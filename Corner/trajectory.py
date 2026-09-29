"""저장한 ㄱ자 전환 경로를 동일한 보간 규칙으로 검증하고 재생한다."""

import json

import numpy as np

from kinematics.anchoring import TipAnchor
from Corner.scene import OUTPUT


def smooth_fraction(value):
    r"""구간 양 끝에서 속도가 영이 되는 보간 비율을 구한다.

    $$s(u)=3u^2-2u^3$$
    """
    # 끝점에서 기울기가 영인 삼차 보간: $$s(u)=3u^2-2u^3$$
    return value * value * (3 - 2 * value)


class Trajectory:
    """영상과 검사가 공유하는 유한 관절 경로를 읽는다."""

    def __init__(self, directory=OUTPUT):
        """저장 배열의 길이·수치·시간 순서와 고정 팁을 검사한다."""
        self.directory = directory
        self.meta = json.loads((directory / "trajectory.json").read_text())
        with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
            self.arrays = {key: archive[key].copy() for key in archive.files}
        n = len(self.arrays["time"])
        expected = {"q": (n, 7), "grippers": (n, 2), "anchor": (n, 4, 4),
                    "L": (n, 4, 4), "R": (n, 4, 4), "phase": (n,), "fixed_tip": (n,)}
        if n < 2 or self.arrays["time"].shape != (n,):
            raise ValueError("두 시점 이상의 일차원 시간 배열이 필요합니다.")
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"경로 배열의 형상이 다릅니다: {name}")
        for name, value in self.arrays.items():
            if name != "fixed_tip" and not np.all(np.isfinite(value)):
                raise ValueError(f"경로에 유한하지 않은 값이 있습니다: {name}")
        if np.any(np.diff(self.arrays["time"]) <= 0):
            raise ValueError("경로 시각은 엄격히 증가해야 합니다.")
        if not np.all(np.isin(self.arrays["fixed_tip"], ["tip_L", "tip_R"])):
            raise ValueError("알 수 없는 고정 팁입니다.")
        if np.any(self.arrays["phase"] < 0) or np.any(self.arrays["phase"] >= len(self.meta["phases"])):
            raise ValueError("단계 번호가 경로 설명의 범위를 벗어났습니다.")
        self.duration = float(self.arrays["time"][-1])

    def segment(self, index, fraction):
        r"""지정 구간에서 관절을 보간하고 다음 상태의 고정 팁을 사용한다.

        $$q(s)=(1-s)q_i+sq_{i+1}$$

        고정단 전환 구간은 관절각이 같아야 하며 검증 단계에서 연속성을 확인한다.
        """
        arrays = self.arrays
        next_index = index + 1
        # 몸통 관절 보간: $$q(s)=(1-s)q_i+sq_{i+1}$$
        q = (1 - fraction) * arrays["q"][index] + fraction * arrays["q"][next_index]
        # 개폐 관절 보간: $$g(s)=(1-s)g_i+sg_{i+1}$$
        grippers = (1 - fraction) * arrays["grippers"][index] + fraction * arrays["grippers"][next_index]
        anchor = TipAnchor(str(arrays["fixed_tip"][next_index]), arrays["anchor"][next_index])
        return q, grippers, anchor, int(arrays["phase"][next_index])

    def sample(self, time):
        r"""재생 시각을 구간 비율로 바꿔 저장 관절 경로를 부드럽게 재생한다.

        $$u=(t-t_i)/(t_{i+1}-t_i)$$
        """
        times = self.arrays["time"]
        time = float(np.clip(time, times[0], times[-1]))
        index = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
        # 해당 시간 구간의 정규화 비율: $$u=(t-t_i)/(t_{i+1}-t_i)$$
        fraction = (time - times[index]) / (times[index + 1] - times[index])
        return self.segment(index, smooth_fraction(fraction))
