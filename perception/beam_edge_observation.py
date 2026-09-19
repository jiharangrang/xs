"""횡이동에 사용할 빔 바깥 모서리를 여러 새 깊이 프레임에서 추적한다.
처음에는 폭을 확인하고 이후에는 같은 모서리와 빔이 놓인 쪽을 확인한다.
"""

from dataclasses import dataclass, replace
import time

import numpy as np

from perception.beam import BeamDetectionError, estimate_beam
from perception.beam_observation import (
    BeamObservation, ObservationSettings, collect_observations, combine_estimates,
)
from perception.depth import plane_points


@dataclass(frozen=True)
class EdgeReference:
    """같은 모서리를 다시 찾기 위한 점·길이 방향·바깥 방향·평면과 최초 폭이다."""

    point_m: np.ndarray
    axis: np.ndarray
    outward: np.ndarray
    normal: np.ndarray
    plane_offset_m: float
    width_m: float

    def transformed(self, rotation, translation):
        r"""모서리와 평면을 같은 강체 변환으로 다른 좌표계에 표현한다.

        $$p'=Rp+t,\quad n'=Rn,\quad d'=d-n'^Tt$$
        """
        # 모서리의 기준점을 변환: $$p'=Rp+t$$
        point = rotation @ self.point_m + translation
        # 법선의 좌표계를 변환: $$n'=Rn$$
        normal = rotation @ self.normal
        # 원점 변화에 따른 평면 상수: $$d'=d-n'^Tt$$
        offset = self.plane_offset_m - normal @ translation
        # 모서리 길이 방향을 변환: $$a'=Ra$$
        axis = rotation @ self.axis
        # 바깥 방향을 변환: $$u'=Ru$$
        outward = rotation @ self.outward
        return EdgeReference(point, axis, outward, normal, float(offset), self.width_m)


@dataclass(frozen=True)
class EdgeObservation(BeamObservation):
    """확인된 모서리 위치와 바깥 방향 및 프레임 간 횡방향 편차를 담는다."""

    edge_point_m: np.ndarray
    axis: np.ndarray
    outward: np.ndarray
    edge_spread_m: float
    single_edge: bool

    def reference(self):
        """현재 관측을 다음 모서리 추적에 사용할 기준으로 반환한다."""
        return EdgeReference(self.edge_point_m.copy(), self.axis.copy(), self.outward.copy(),
                             self.normal.copy(), self.plane_offset_m, self.width_m)

    def as_dict(self):
        """영상 배열 없이 모서리 좌표와 측정 품질을 직렬화한다."""
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()}


def select_edge(estimate, profile, *, outward_hint, reference, settings):
    r"""진행할 쪽의 모서리를 선택하고 관측 평면이 그 안쪽에 있는지 확인한다.

    $$u=\frac{n\times a}{\|n\times a\|},\quad s(p)=u^T(p-p_e)$$

    바깥 방향은 고정턱 쪽으로 정하며 빔 표면은 음의 횡좌표 쪽에 있어야 한다.
    """
    two_edges = (estimate.status == "two_edges" and estimate.width_m is not None
                 and abs(estimate.width_m - settings.beam_width_m) <= settings.width_tolerance_m)
    if reference is None and not two_edges:
        raise BeamDetectionError("첫 관측에서는 빔의 두 모서리와 폭을 확인해야 합니다.")
    if reference is not None:
        if (estimate.normal @ reference.normal < np.cos(np.deg2rad(5))
                or abs(estimate.normal @ reference.point_m + estimate.plane_offset_m) > .010):
            raise BeamDetectionError("처음 확인한 빔 평면을 다시 찾고 있습니다.")
    lines = estimate.edge_lines_m if reference is None else estimate.boundary_lines_m
    if lines is None:
        lines = estimate.edge_lines_m
    if lines is None or not len(lines):
        raise BeamDetectionError("진행할 쪽의 빔 모서리를 기다립니다.")
    pixels = np.argwhere(estimate.plane_mask)
    pixels = pixels[::max(1, len(pixels) // 1000), ::-1]
    surface = plane_points(pixels, profile, estimate.normal, estimate.plane_offset_m)
    candidates = []
    for line in lines:
        # 모서리의 관측 길이 벡터: $$v=p_1-p_0$$
        vector = line[1] - line[0]
        length = np.linalg.norm(vector)
        if length < .025:
            continue
        # 모서리의 단위 길이 방향: $$a=v/\|v\|$$
        axis = vector / length
        # 빔 폭의 바깥 방향 후보: $$u=n\times a$$
        outward = np.cross(estimate.normal, axis)
        # 폭 방향을 단위 벡터로 변환: $$u=u/\|u\|$$
        outward /= np.linalg.norm(outward)
        if outward @ outward_hint < 0:
            axis, outward = -axis, -outward
        point = line.mean(axis=0)
        if reference is not None:
            if abs(axis @ reference.axis) < np.cos(np.deg2rad(8)):
                continue
            # 예측한 같은 모서리와의 횡방향 차이: $$e=|u^T(p_e-p_{ref})|$$
            score = abs(float(outward @ (point - reference.point_m)))
            if score > .012:
                continue
        else:
            # 바깥쪽 모서리에 작은 선택 점수를 부여: $$e=-u^Tp_e$$
            score = -float(outward @ point)
        # 관측한 표면의 모서리 기준 횡좌표: $$s_i=u^T(p_i-p_e)$$
        sidedness = (surface - point) @ outward
        if np.mean(sidedness <= .002) < .85:
            continue
        candidates.append((score, point, axis, outward))
    if not candidates:
        raise BeamDetectionError("같은 모서리와 빔이 놓인 쪽을 다시 확인하고 있습니다.")
    _, point, axis, outward = min(candidates, key=lambda item: item[0])
    width = estimate.width_m if two_edges else reference.width_m
    return replace(estimate, width_m=width), point, axis, outward, not two_edges


def combine_edges(samples, timestamps, settings):
    r"""모서리 위치와 방향이 안정된 여러 관측을 하나로 묶는다.

    $$\sigma_e=\max_i|u^T(p_i-\bar p)|$$
    """
    base = combine_estimates([sample[0] for sample in samples], timestamps, settings)
    points = np.array([sample[1] for sample in samples])
    axes = np.array([sample[2] for sample in samples])
    # 관측 구간 중심의 평균 위치: $$\bar p=\operatorname{mean}(p_i)$$
    point = points.mean(axis=0)
    # 여러 모서리의 평균 단위 방향: $$a=\sum_i a_i/\|\sum_i a_i\|$$
    axis = axes.sum(axis=0) / np.linalg.norm(axes.sum(axis=0))
    # 평면과 모서리 방향으로 바깥 축을 구성: $$u=n\times a$$
    outward = np.cross(base.normal, axis)
    # 횡방향 간격 계산에 사용할 단위 벡터: $$u=u/\|u\|$$
    outward /= np.linalg.norm(outward)
    # 프레임 간 모서리의 최대 횡방향 편차: $$\sigma_e=\max_i|u^T(p_i-\bar p)|$$
    spread = float(np.max(np.abs((points - point) @ outward)))
    if spread > .002 or np.min(axes @ axis) < np.cos(np.deg2rad(2)):
        raise BeamDetectionError("프레임 사이 모서리 위치가 달라 다시 관측합니다.")
    return EdgeObservation(**vars(base), edge_point_m=point, axis=axis, outward=outward,
                           edge_spread_m=spread, single_edge=any(sample[4] for sample in samples))


def observe_beam_edge(reader, after_s, *, outward_hint, reference=None, settings=None,
                      cancelled=lambda: False, clock=time.monotonic, detector=estimate_beam):
    """정지 후 새 영상에서 확인한 모서리만 반환하며 잠깐의 검출 실패는 재관측한다."""
    settings = settings if settings is not None else ObservationSettings()

    def select(estimate, profile):
        """현재 기준 모서리와 같은 후보를 고른다."""
        return select_edge(estimate, profile, outward_hint=outward_hint, reference=reference, settings=settings)

    return collect_observations(reader, after_s, settings=settings, cancelled=cancelled,
                                clock=clock, detector=detector, select=select, combine=combine_edges)
