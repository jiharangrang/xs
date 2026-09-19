"""고정턱 안쪽 옆면이 빔 밖으로 지정 간격만큼 빠지는 작은 횡이동을 계산한다.
카메라의 빔 모서리와 CAD 고정턱을 비교하고 기존 직선 IK를 재사용한다.
"""

from dataclasses import dataclass

import numpy as np

from kinematics.anchoring import TipAnchor
from kinematics.gripper_geometry import GripperGeometry
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles
from planning.linear_motion import plan_linear_motion


@dataclass(frozen=True)
class ExitSettings:
    """외측 여유와 작은 횡이동의 범위 및 빔 아래 관측 조건을 지정한다."""

    clearance_m: float = .010
    tolerance_m: float = .001
    max_step_m: float = .005
    max_total_m: float = .060
    max_joint_step_deg: float = 3.
    min_vertical_gap_m: float = .015
    max_entry_gap_m: float = .035
    max_tilt_deg: float = 3.

    def __post_init__(self):
        """거리와 각도 기준의 유효 범위를 검사한다."""
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("횡이동 설정은 유한한 양수여야 합니다.")
        if self.tolerance_m >= self.clearance_m or self.min_vertical_gap_m >= self.max_entry_gap_m:
            raise ValueError("횡이동 간격과 높이 범위를 확인해 주세요.")


@dataclass(frozen=True)
class ExitMeasurement:
    """고정턱의 옆 간격과 전체 그리퍼의 빔 아래 간격을 구분한다."""

    clearance_m: float
    vertical_gap_m: float


@dataclass(frozen=True)
class ExitStep:
    """작은 횡이동의 관절 목표와 월드 변위를 담는다."""

    q_rad: np.ndarray
    displacement_world_m: np.ndarray
    distance_m: float


class BeamExitPlanner:
    """고정턱이 있는 쪽으로 움직이며 팁의 방향과 빔 평면까지 높이를 유지한다."""

    def __init__(self, *, settings=None):
        """공통 메시 기하와 단일 초기값 직선 IK를 준비한다."""
        self.settings = settings if settings is not None else ExitSettings()
        self.geometry = GripperGeometry()
        self.solver = InverseKinematics(settings=IKSettings(starts=1))

    def outward_hint(self, q_rad, grippers_deg):
        """실측 자세에서 고정턱이 있는 쪽의 카메라 방향을 반환한다."""
        self.geometry.update(q_rad, grippers_deg)
        return self.geometry.outward_camera()

    def measure(self, q_rad, grippers_deg, observation):
        r"""고정턱의 가장 안쪽 점과 모서리 사이의 부호 있는 옆 간격을 구한다.

        $$g_{side}=\min_i u^T(p_{lip,i}-p_e),\quad g_{below}=\min_j(n^Tp_j+d)$$

        옆 간격이 음수면 고정턱이 빔과 겹치며, 열린 턱은 높이 계산에만 포함한다.
        """
        self.geometry.update(q_rad, grippers_deg)
        if observation.outward @ self.geometry.outward_camera() <= 0:
            raise ValueError("관측한 바깥 방향이 고정턱 방향과 다릅니다.")
        lip = self.geometry.points_camera(fixed_lip=True)
        # 고정턱 안쪽 면과 둥근 모서리의 최소 옆 간격: $$g_{side}=\min_i u^T(p_{lip,i}-p_e)$$
        clearance = float(np.min((lip - observation.edge_point_m) @ observation.outward))
        points = self.geometry.points_camera()
        # 두 턱 모두의 빔 아래 최소 간격: $$g_{below}=\min_j(n^Tp_j+d)$$
        vertical = float(np.min(points @ observation.normal + observation.plane_offset_m))
        if not np.all(np.isfinite([clearance, vertical])):
            raise ValueError("관측한 옆 간격과 높이가 유효하지 않습니다.")
        return ExitMeasurement(clearance, vertical)

    def plan(self, q_rad, grippers_deg, observation):
        r"""현재 옆 간격에 맞는 작은 평면 내 횡이동을 계산한다.

        $$s=\min(s_{max},g_{target}-g_{side}),\quad \Delta p_W=sR_{WC}u$$
        """
        q_rad = as_joint_angles(q_rad)
        measured = self.measure(q_rad, grippers_deg, observation)
        if measured.clearance_m >= self.settings.clearance_m - self.settings.tolerance_m:
            raise ValueError("이미 고정턱의 목표 옆 간격에 도달했습니다.")
        start = self.solver.fk.forward(q_rad)
        camera = self.solver.fk.depth_camera_pose(q_rad)
        anchor = TipAnchor("tip_L", start.T_world_tip_L)
        # 남은 옆 간격에 맞춰 한 번의 이동량 제한: $$s=\min(s_{max},g_{target}-g_{side})$$
        distance = min(self.settings.max_step_m, self.settings.clearance_m - measured.clearance_m)
        for scale in (1., .5, .25):
            # 관측한 빔 폭 방향의 월드 변위: $$\Delta p_W=ksR_{WC}u$$
            displacement = scale * distance * (camera[:3, :3] @ observation.outward)
            result = plan_linear_motion(q_rad, anchor, displacement, steps=5, solver=self.solver)
            if not result.success:
                continue
            # 실제 전송할 두 관절 목표 사이 최대 변화: $$\Delta q_{max}=\max_j|q_{d,j}-q_{c,j}|180/\pi$$
            change = float(np.max(np.abs(np.rad2deg(result.q_path_rad[-1] - q_rad))))
            if change > self.settings.max_joint_step_deg:
                continue
            # 작은 이동의 시작·끝 목표 전체 구간을 표본 검사: $$q(t)=(1-t)q_c+tq_d$$
            samples = [(1 - t) * q_rad + t * result.q_path_rad[-1] for t in np.linspace(0, 1, 9)]
            # 표본 자세의 빔 법선 방향 변위: $$h(t)=n_C^TR_{WC}^T(p_{tip}(q(t))-p_{tip}(q_c))$$
            heights = [observation.normal @ (camera[:3, :3].T @
                       (self.solver.fk.forward(q).T_world_tip_R[:3, 3] - start.T_world_tip_R[:3, 3])) for q in samples]
            if max(abs(value) for value in heights) > .0005:
                continue
            # 실제 계획한 작은 이동의 길이: $$s_{step}=\|\Delta p_W\|$$
            step_distance = float(np.linalg.norm(displacement))
            return ExitStep(result.q_path_rad[-1].copy(), displacement, step_distance)
        raise ValueError("작은 관절 변화로 높이를 유지하는 횡이동 자세를 찾지 못했습니다.")
