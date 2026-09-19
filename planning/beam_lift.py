"""실측 빔 평면과 현재 그리퍼 형상으로 남은 간격과 작은 상승 자세를 계산한다.
기존 직선 IK와 정면 보정 계획기를 재사용하며 실물 모터를 움직이지 않는다.
"""

from dataclasses import dataclass

import numpy as np

from kinematics.gripper_geometry import GripperGeometry
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles
from planning.beam_alignment import BeamAlignmentPlanner, tilt_degrees, unit_normal
from planning.vertical_motion import plan_vertical_step


@dataclass(frozen=True)
class LiftSettings:
    """그리퍼와 빔 사이의 목표 간격 및 한 번과 전체 실행의 상승 한도를 지정한다."""

    gap_m: float = .025
    tolerance_m: float = .002
    max_step_m: float = .005
    max_total_m: float = .060
    max_joint_step_deg: float = 3.
    alignment_tolerance_deg: float = 1.

    def __post_init__(self):
        """거리·각도 기준이 유한한 양수이며 목표 간격이 허용오차보다 큰지 검사한다."""
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()) or self.gap_m <= self.tolerance_m:
            raise ValueError("상승 설정은 유한한 양수이고 목표 간격은 허용오차보다 커야 합니다.")


@dataclass(frozen=True)
class LiftStep:
    """관절 목표와 계획한 상승량을 담으며 자세만 보정하는 경우 상승량은 영이다."""

    q_rad: np.ndarray
    distance_m: float
    gap_before_m: float
    predicted_gap_m: float
    kind: str


class BeamLiftPlanner:
    """현재 열린 그리퍼의 가장 가까운 점을 기준으로 빔과의 간격을 계산한다."""

    def __init__(self, *, settings=None):
        """공통 모델·직선 IK와 정면 보정 솔버를 한 번 준비한다."""
        self.settings = settings if settings is not None else LiftSettings()
        self.solver = InverseKinematics(settings=IKSettings(starts=1))
        self.aligner = BeamAlignmentPlanner()
        self.geometry = GripperGeometry()

    def gap(self, q_rad, grippers_deg, normal, plane_offset_m) -> float:
        r"""실측 평면과 그리퍼 메시 꼭짓점 사이 최소 간격을 카메라 좌표에서 계산한다.

        $$g=\min_i(n_C^Tp_{C,i}+d),\quad p_{C,i}=R_{WC}^T(p_{W,i}-p_{WC})$$

        카메라 장착 CAD를 사용하는 간격 추정이며 빔 옆면·다른 링크와의 충돌 검사는 아니다.
        """
        q_rad = as_joint_angles(q_rad)
        normal = unit_normal(normal)
        if not np.isfinite(plane_offset_m) or plane_offset_m <= 0:
            raise ValueError("유효한 빔 평면 거리가 필요합니다.")
        self.geometry.update(q_rad, grippers_deg)
        points = self.geometry.points_camera()
        # 카메라 쪽을 양수로 하는 최소 평면 간격: $$g=\min_i(n_C^Tp_{C,i}+d)$$
        return float(np.min(points @ normal + plane_offset_m))

    def plan(self, q_rad, grippers_deg, normal, plane_offset_m) -> LiftStep:
        r"""새 관측의 남은 간격에 맞춰 자세를 유지하는 작은 직선 상승을 계산한다.

        $$h=\min(h_{max},g-g_{target}),\quad \Delta p_W=-h R_{WC}n_C$$

        기울기가 다시 커졌으면 먼저 현재 팁 위치에서 정면을 맞춘다.
        """
        q_rad = as_joint_angles(q_rad)
        normal = unit_normal(normal)
        gap = self.gap(q_rad, grippers_deg, normal, plane_offset_m)
        if tilt_degrees(normal) > self.settings.alignment_tolerance_deg:
            step = self.aligner.plan(q_rad, normal, q_rad)
            return LiftStep(step.q_rad, 0., gap, gap, "alignment")
        if gap <= self.settings.gap_m + self.settings.tolerance_m:
            raise ValueError("이미 목표 간격에 도달해 추가 상승이 필요하지 않습니다.")
        # 한 번에 줄일 간격을 제한: $$h=\min(h_{max},g-g_{target})$$
        height = min(self.settings.max_step_m, gap - self.settings.gap_m)
        result = plan_vertical_step(q_rad, normal, height, solver=self.solver, steps=5,
                                    max_joint_step_deg=self.settings.max_joint_step_deg)
        if result is not None:
            target, distance = result
            # 계획 변위를 반영한 이동 후 간격: $$g_{next}=g-s$$
            predicted_gap = gap - distance
            return LiftStep(target, distance, gap, predicted_gap, "lift")
        raise ValueError("작은 관절 변화 범위에서 상승 자세를 찾지 못했습니다.")
