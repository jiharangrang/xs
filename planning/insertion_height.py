"""실물에서 확인한 빔 평면 거리로 높이 오차를 계산하고 높이·정면을 함께 보정한다.
그리퍼 형상으로 계산한 주변 여유는 목표 거리와 분리해 표시용으로만 제공한다.
"""

from dataclasses import dataclass, field

import numpy as np

from kinematics.gripper_geometry import GripperGeometry
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles
from kinematics.mesh_sections import strip_vertices
from planning.beam_alignment import tilt_degrees, unit_normal
from planning.vertical_motion import solve_height_alignment_target
from planning.height_reference import DEFAULT_DEPTH_REFERENCE_PATH, HeightReference, load_height_reference


def load_target_depth(path=DEFAULT_DEPTH_REFERENCE_PATH):
    """도착 기준으로 보관한 원래 카메라 거리 값을 반환한다."""
    return load_height_reference(path).depth_m


@dataclass(frozen=True)
class InsertionHeightSettings:
    """실측 목표 깊이와 허용오차·보정 크기 및 표시용 빔 두께를 지정한다."""

    reference: HeightReference = field(default_factory=load_height_reference)
    flange_thickness_m: float = .008
    tolerance_m: float = .0005
    max_step_m: float = .005
    fine_step_m: float = .0005
    fine_zone_m: float = .005
    alignment_tolerance_deg: float = 1.
    alignment_gain: float = .6
    max_camera_step_deg: float = 2.

    def __post_init__(self):
        """계산에 사용할 거리 설정이 유효한 양수인지 검사한다."""
        if not isinstance(self.reference, HeightReference):
            raise ValueError("실물 도착 자세의 높이 기준이 필요합니다.")
        if any(not np.isfinite(value) or value <= 0 for name, value in vars(self).items() if name != "reference"):
            raise ValueError("높이 보정 설정은 유한한 양수여야 합니다.")
        if self.alignment_gain > 1:
            raise ValueError("정면 보정 이득은 1 이하여야 합니다.")
        if self.fine_step_m > self.max_step_m or self.fine_zone_m < self.fine_step_m:
            raise ValueError("높이 보정의 접근·정밀 이동 크기를 확인해 주세요.")

    @property
    def target_depth_m(self):
        """기존 표시와 저장값 확인에 사용할 도착 당시 원본 거리를 반환한다."""
        return self.reference.depth_m


@dataclass(frozen=True)
class InsertionHeightMeasurement:
    """실측 깊이·목표 거리·높이 오차와 표시용 위·아래·옆 간격을 담는다."""

    upper_clearance_m: float
    lower_clearance_m: float | None
    side_clearance_m: float
    remaining_m: float
    depth_m: float
    target_depth_m: float


@dataclass(frozen=True)
class InsertionHeightStep:
    """높이 보정 관절 목표와 부호가 있는 이동량·정밀 구간 여부를 담는다."""

    q_rad: np.ndarray
    distance_m: float
    fine: bool
    kind: str = "lift"


class InsertionHeightPlanner:
    """고정턱이 빠진 횡위치에서 실물로 확인한 거리와 정면을 함께 맞춘다."""

    def __init__(self, *, settings=None):
        """높이 기준 형상과 공통 IK를 준비한다."""
        self.settings = settings if settings is not None else InsertionHeightSettings()
        self.geometry = GripperGeometry()
        self.solver = InverseKinematics(settings=IKSettings(starts=1))

    def outward_hint(self, q_rad, grippers_deg):
        """현재 고정턱 쪽을 모서리 관측의 바깥 방향으로 제공한다."""
        self.geometry.update(q_rad, grippers_deg)
        return self.geometry.outward_camera()

    def height_error(self, q_rad, normal, depth_m):
        r"""현재 팁 기준의 높이 오차와 자세에 맞는 카메라 목표 거리를 계산한다.

        $$e_h=d+n^Tr-g_*,\quad r=R_{WC}^T(p_{tip}-p_{WC})$$

        e_h가 양수일 때만 상승이 필요하다. 카메라 회전에 따른 원점 거리 변화는 제거한다.
        """
        if not np.isfinite(depth_m) or depth_m <= 0:
            raise ValueError("유효한 빔 평면 거리가 필요합니다.")
        q_rad = as_joint_angles(q_rad)
        camera = self.solver.fk.depth_camera_pose(q_rad)
        tip = self.solver.fk.forward(q_rad).T_world_tip_R[:3, 3]
        # 현재 그리퍼 기준점을 깊이 카메라 좌표로 표현: $$r=R_{WC}^T(p_{tip}-p_{WC})$$
        point = camera[:3, :3].T @ (tip - camera[:3, 3])
        goal = self.settings.reference.camera_goal(normal, point)
        # 같은 지점의 높이를 비교한 부호 있는 오차: $$e_h=d-d_{goal}$$
        error = float(depth_m - goal)
        return error, goal

    def measure(self, q_rad, grippers_deg, observation):
        r"""실측 평면 거리와 저장한 목표 거리의 차이를 높이 보정 오차로 계산한다.

        $$h=(d+n^Tr)-(d_*+n_*^Tr_*)$$

        실측 도착 자세와 현재 자세에서 동일한 그리퍼 기준점의 높이를 비교한다.
        양수이면 상승하고 음수이면 낮춘다. CAD 여유는 표시만 하며 도착 판정에 쓰지 않는다.
        """
        self.geometry.update(q_rad, grippers_deg)
        normal, outward = unit_normal(observation.normal), observation.outward
        if not np.isfinite(observation.plane_offset_m) or observation.plane_offset_m <= 0:
            raise ValueError("유효한 빔 깊이가 필요합니다.")
        depth = float(observation.plane_offset_m)
        remaining, goal_depth = self.height_error(q_rad, normal, depth)
        lip = self.geometry.points_camera(fixed_lip=True)
        # 고정턱 끝 전체가 빔 모서리 밖에 있는 최소 거리: $$c_s=\min_i u^T(p_{lip,i}-p_e)$$
        side = float(np.min((lip - observation.edge_point_m) @ outward))
        # 빔 윗면보다 고정턱 접촉면이 높은 최소 여유: $$c_u=-\max_i(n^Tp_{lip,i}+d)-t_f$$
        upper = -float(np.max(lip @ normal + observation.plane_offset_m)) - self.settings.flange_thickness_m
        # 관측한 바깥 모서리의 횡좌표: $$b=u^Tp_e$$
        edge = float(outward @ observation.edge_point_m)
        lower_points = strip_vertices(self.geometry.lower_triangles_camera(), outward,
                                       edge - observation.width_m, edge)
        # 실제 빔 폭과 겹치는 아래쪽 형상의 최소 여유: $$c_l=\min_j(n^Tp_{body,j}+d)$$
        lower = float(np.min(lower_points @ normal + observation.plane_offset_m)) if len(lower_points) else None
        result = InsertionHeightMeasurement(upper, lower, side, remaining, depth, goal_depth)
        if not all(value is None or np.isfinite(value) for value in vars(result).values()):
            raise ValueError("삽입 높이 계산에 유효하지 않은 관측이 있습니다.")
        return result

    def correction(self, remaining, normal):
        r"""높이 오차와 빔 법선을 작은 높이 변위와 카메라 회전으로 바꾼다.

        $$s=\operatorname{sgn}(h)\min(|h|,s_{limit}),\quad
        \omega=\min(g\theta,\alpha_{max})\hat a$$

        허용 범위에 든 성분은 영으로 두어 상승과 삽입에서 같은 보정 크기를 재사용한다.
        """
        normal = unit_normal(normal)
        tilt = tilt_degrees(normal)
        height_reached = abs(remaining) <= self.settings.tolerance_m
        alignment_reached = tilt <= self.settings.alignment_tolerance_deg
        fine = abs(remaining) <= self.settings.fine_zone_m + self.settings.fine_step_m
        limit = self.settings.fine_step_m if fine else self.settings.max_step_m
        # 목표를 향한 부호 있는 한 번의 높이 보정량: $$s=\operatorname{sgn}(h)\min(|h|,s_{limit})$$
        distance = 0. if height_reached else float(np.sign(remaining) * min(abs(remaining), limit))
        rotation_vector = np.zeros(3)
        if not alignment_reached:
            # 광축에서 관측 빔 정면으로 향하는 회전축: $$a=e_z\times(-n)$$
            axis = np.cross([0., 0., 1.], -normal)
            # 회전축 정규화: $$\hat a=a/\|a\|$$
            axis /= np.linalg.norm(axis)
            # 이득을 적용한 보정각을 라디안으로 변환: $$\alpha=\min(g\theta,\alpha_{max})\pi/180$$
            angle = np.deg2rad(min(self.settings.alignment_gain * tilt, self.settings.max_camera_step_deg))
            # 카메라 좌표의 정면 보정 회전 벡터: $$\omega=\alpha\hat a$$
            rotation_vector = angle * axis
        return distance, rotation_vector, fine

    def plan(self, q_rad, grippers_deg, observation):
        """같은 관측에서 구한 높이와 정면 보정을 하나의 관절 목표로 변환한다."""
        q_rad = as_joint_angles(q_rad)
        measured = self.measure(q_rad, grippers_deg, observation)
        if abs(measured.remaining_m) <= self.settings.tolerance_m:
            raise ValueError("목표 높이에 도달해 추가 보정이 필요하지 않습니다.")
        normal = unit_normal(observation.normal)
        distance, rotation_vector, fine = self.correction(measured.remaining_m, normal)
        result = solve_height_alignment_target(q_rad, normal, distance, rotation_vector, solver=self.solver)
        if result is None:
            raise ValueError("현재 관절 범위에서 높이·정면 보정 목표의 IK 해를 찾지 못했습니다.")
        target, actual_distance = result
        kind = "lift" if actual_distance > 0 else "lower" if actual_distance < 0 else "alignment"
        return InsertionHeightStep(target, actual_distance, fine, kind)
