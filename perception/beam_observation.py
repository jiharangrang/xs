"""정지 이후의 새 깊이 프레임들을 모아 빔 법선의 일관성을 확인한다.
단일 프레임 실패는 다시 관측하며 오래된 영상이나 빔을 확인하지 못한 평면은 반환하지 않는다.
"""

from dataclasses import dataclass
import time

import numpy as np

from perception.beam import BeamDetectionError, estimate_beam


@dataclass(frozen=True)
class ObservationSettings:
    """빔 확인과 여러 프레임의 관측 품질에 사용할 기준이다."""

    frames: int = 3
    timeout_s: float = 3.
    max_age_s: float = .75
    beam_width_m: float = .070
    width_tolerance_m: float = .015
    max_plane_rms_m: float = .002
    max_normal_spread_deg: float = 1.

    def __post_init__(self):
        """프레임 수와 품질 기준의 유효 범위를 검사한다."""
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 2:
            raise ValueError("연속 관측 프레임 수는 2 이상이어야 합니다.")
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("관측 기준은 유한한 양수여야 합니다.")


@dataclass(frozen=True)
class BeamObservation:
    """여러 새 프레임에서 일치한 법선과 관측 시각·거리·품질을 담는다."""

    normal: np.ndarray
    plane_offset_m: float
    width_m: float
    plane_rms_m: float
    normal_spread_deg: float
    first_frame_s: float
    last_frame_s: float
    frames: int

    def as_dict(self):
        """로그와 화면에 전달할 수 있는 기본 자료형으로 변환한다."""
        return {**vars(self), "normal": self.normal.tolist()}


def combine_estimates(estimates, timestamps, settings) -> BeamObservation:
    r"""법선을 평균하고 프레임 간 각도 차이가 작은 관측만 반환한다.

    $$\bar n=\frac{\sum_i n_i}{\|\sum_i n_i\|},\quad
    \sigma_n=\max_i\cos^{-1}(n_i^T\bar n)180/\pi$$
    """
    # 여러 프레임 법선의 벡터 합: $$s=\sum_i n_i$$
    normal = np.sum([estimate.normal for estimate in estimates], axis=0)
    # 평균 방향의 단위 법선: $$\bar n=s/\|s\|$$
    normal /= np.linalg.norm(normal)
    # 평균 방향에서 가장 멀리 벗어난 관측 각도: $$\sigma_n=\max_i\cos^{-1}(n_i^T\bar n)180/\pi$$
    spread = float(np.max(np.rad2deg(np.arccos(np.clip(np.array([e.normal for e in estimates]) @ normal, -1, 1)))))
    if not np.isfinite(spread) or spread > settings.max_normal_spread_deg:
        raise BeamDetectionError("프레임 사이 빔 기울기가 달라 안정된 관측을 기다립니다.")
    return BeamObservation(normal, float(np.median([e.plane_offset_m for e in estimates])),
                           float(np.median([e.width_m for e in estimates])),
                           max(e.plane_rms_m for e in estimates), spread,
                           min(timestamps), max(timestamps), len(estimates))


def observe_beam(reader, after_s: float, *, settings=None, cancelled=lambda: False,
                 clock=time.monotonic, detector=estimate_beam) -> BeamObservation:
    """지정 시각 이후의 새 프레임들에서 폭으로 빔을 확인하고 안정된 법선을 구한다."""
    settings = settings if settings is not None else ObservationSettings()

    def select(estimate, profile):
        """두 모서리 사이 폭으로 빔인지 확인한다."""
        if (estimate.status != "two_edges" or estimate.width_m is None
                or abs(estimate.width_m - settings.beam_width_m) > settings.width_tolerance_m):
            raise BeamDetectionError("빔의 두 모서리와 폭을 다시 확인하고 있습니다.")
        return estimate

    return collect_observations(reader, after_s, settings=settings, cancelled=cancelled,
                                clock=clock, detector=detector, select=select, combine=combine_estimates)


def collect_observations(reader, after_s, *, settings, cancelled, clock, detector, select, combine):
    """새 프레임·평면 품질 검사를 공유하고 단계별 모서리 선택과 묶음 계산을 호출한다."""
    deadline = clock() + settings.timeout_s
    estimates, timestamps = [], []
    last_stamp = None
    reason = "새 빔 영상을 기다립니다."
    while clock() < deadline:
        if cancelled():
            raise BeamDetectionError("정면 보정 관측을 중지했습니다.")
        frame = reader.read(timeout_ms=500)
        if (frame.received_monotonic_s <= after_s or clock() - frame.received_monotonic_s > settings.max_age_s
                or frame.depth_timestamp_us == last_stamp):
            continue
        last_stamp = frame.depth_timestamp_us
        try:
            estimate = detector(frame.depth_m, reader.info["depth"])
            if estimate.plane_rms_m > settings.max_plane_rms_m or estimate.inlier_fraction < .5:
                raise BeamDetectionError("빔 평면의 깊이 품질이 부족해 재관측합니다.")
            estimates.append(select(estimate, reader.info["depth"]))
            timestamps.append(frame.received_monotonic_s)
            if len(estimates) >= settings.frames:
                observation = combine(estimates, timestamps, settings)
                if clock() - observation.last_frame_s > settings.max_age_s:
                    raise BeamDetectionError("처리한 영상이 오래되어 다시 관측합니다.")
                return observation
        except BeamDetectionError as error:
            reason = str(error)
            estimates.clear()
            timestamps.clear()
    raise BeamDetectionError(reason)
