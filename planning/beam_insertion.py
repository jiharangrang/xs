"""4단계 출발 횡위치까지 돌아갈 삽입량을 현재 실측 자세에서 계산한다.
실측 목표 높이와 정면 보정을 함께 적용하고 공통 기구학·모터 목표 형식을 재사용한다.
"""

from dataclasses import dataclass

import numpy as np

from kinematics.joints import as_joint_angles
from planning.beam_alignment import tilt_degrees, unit_normal
from planning.insertion_height import InsertionHeightPlanner
from planning.vertical_motion import solve_camera_tip_target


@dataclass(frozen=True)
class InsertionSettings:
    """출발 횡위치와 완료 높이의 허용오차 및 한 번에 접근할 거리를 지정한다."""

    tolerance_m: float = .001
    max_step_m: float = .003
    fine_step_m: float = .001
    fine_zone_m: float = .004
    arrival_height_tolerance_m: float = .002

    def __post_init__(self):
        """삽입 거리와 완료 허용오차가 유한한 양수인지 확인한다."""
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("삽입 거리 설정은 유한한 양수여야 합니다.")
        if self.fine_step_m > self.max_step_m or self.fine_zone_m < self.fine_step_m:
            raise ValueError("삽입의 접근·정밀 이동 크기를 확인해 주세요.")


@dataclass(frozen=True)
class InsertionMeasurement:
    """출발 횡위치까지의 오차와 실측 높이·정면 오차 및 고정턱 겹침을 담는다."""

    remaining_m: float
    height_error_m: float
    depth_m: float
    goal_depth_m: float
    tilt_deg: float
    overlap_m: float
    outward: np.ndarray
    rgb_center_camera_m: np.ndarray


@dataclass(frozen=True)
class InsertionStep:
    """횡이동과 높이·정면 보정이 합쳐진 관절 목표와 각 성분의 계획 거리를 담는다."""

    q_rad: np.ndarray
    distance_m: float
    lateral_distance_m: float
    fine: bool


class BeamInsertionPlanner:
    """4단계 출발 횡위치를 목표로 삼고 현재 빔을 관측하며 옆으로 삽입한다."""

    def __init__(self, *, settings=None, height_settings=None, reference=None):
        """최종 상승의 형상·IK·보정 설정을 그대로 재사용한다."""
        self.settings = settings if settings is not None else InsertionSettings()
        self.height = InsertionHeightPlanner(settings=height_settings)
        self.geometry, self.solver = self.height.geometry, self.height.solver
        self.reference = reference

    def outward_hint(self, q_rad, grippers_deg):
        """동일한 고정턱 기준으로 같은 쪽 빔 모서리를 선택한다."""
        return self.height.outward_hint(q_rad, grippers_deg)

    def measure(self, q_rad, grippers_deg, observation):
        r"""기록된 출발 횡위치까지의 오차와 현재 빔의 높이·정면 오차를 구한다.

        $$e_y=u_W^T(p_{tip,W}-p_{start,W}),\quad e_h=d+n^Tr-g_*,\quad
        o=-\min_i u^T(p_{lip,i}-p_e)$$

        u는 빔 바깥 방향이고 e_y가 양수이면 그 반대인 빔 안쪽으로 이동한다.
        o는 고정턱의 영상·CAD 기준 겹침으로 실제 파지력이나 접촉을 뜻하지 않는다.
        """
        if self.reference is None:
            raise ValueError("삽입 목표로 사용할 4단계 출발 기준이 필요합니다.")
        self.geometry.update(q_rad, grippers_deg)
        normal = unit_normal(observation.normal)
        depth, width = observation.plane_offset_m, observation.width_m
        if not np.all(np.isfinite([depth, width])) or depth <= 0 or width <= 0:
            raise ValueError("삽입에는 유효한 빔 깊이와 폭이 필요합니다.")
        outward = np.asarray(observation.outward, dtype=float)
        point = np.asarray(observation.edge_point_m, dtype=float)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (outward, point)):
            raise ValueError("유효한 빔 모서리 좌표와 횡방향이 필요합니다.")
        # 관측 법선 성분을 제거한 평면 내 횡축: $$u'=u-(u^Tn)n$$
        outward = outward - float(outward @ normal) * normal
        if np.linalg.norm(outward) < 1e-8:
            raise ValueError("빔 평면 위 횡방향을 구할 수 없습니다.")
        # 횡방향 축 정규화: $$u=u'/\|u'\|$$
        outward /= np.linalg.norm(outward)
        if outward @ self.geometry.outward_camera() <= 0:
            raise ValueError("같은 고정턱 쪽 빔 모서리를 다시 확인해 주세요.")
        center = self.geometry.rgb_center_on_plane(normal, depth)
        tip = self.solver.fk.forward(q_rad).T_world_tip_R[:3, 3]
        remaining = self.reference.remaining(tip)
        height_error, goal_depth = self.height.height_error(q_rad, normal, depth)
        lip = self.geometry.points_camera(fixed_lip=True)
        # 고정턱 안쪽 면이 모서리 안으로 들어간 최소 깊이: $$o=-\min_i u^T(p_{lip,i}-p_e)$$
        overlap = -float(np.min((lip - point) @ outward))
        return InsertionMeasurement(remaining, height_error, float(depth), goal_depth, tilt_degrees(normal),
                                    overlap, outward, center)

    def plan(self, q_rad, grippers_deg, observation, *, insert_enabled=True):
        r"""현재 모서리를 향하는 횡이동과 높이·정면 보정을 하나의 목표로 계산한다.

        $$\Delta p_C=-s_yu-s_hn,\quad
        s_y=\operatorname{sgn}(e_y)\min(|e_y|,s_{max})$$

        시작 높이를 맞추기 전에는 횡성분만 영으로 두며 높이·정면 보정은 계속한다.
        """
        q_rad = as_joint_angles(q_rad)
        measured = self.measure(q_rad, grippers_deg, observation)
        if abs(measured.remaining_m) <= self.settings.tolerance_m and abs(measured.height_error_m) <= self.height.settings.tolerance_m:
            raise ValueError("목표 삽입 위치와 높이에 도달해 추가 보정이 필요하지 않습니다.")
        normal = unit_normal(observation.normal)
        height, rotation, height_fine = self.height.correction(measured.height_error_m, normal)
        fine = abs(measured.remaining_m) <= self.settings.fine_zone_m
        limit = self.settings.fine_step_m if fine else self.settings.max_step_m
        lateral = 0.
        if insert_enabled and abs(measured.remaining_m) > self.settings.tolerance_m:
            # 출발 횡위치를 향한 작은 보정량: $$s_y=\operatorname{sgn}(e_y)\min(|e_y|,s_{max})$$
            lateral = float(np.sign(measured.remaining_m) * min(abs(measured.remaining_m), limit))
        if lateral == 0. and height == 0. and not np.any(rotation):
            raise ValueError("목표 삽입 위치에 도달해 추가 보정이 필요하지 않습니다.")
        # 같은 명령에 횡이동과 높이 보정 합성: $$\Delta p_C=-s_yu-s_hn$$
        displacement = -lateral * measured.outward - height * normal
        result = solve_camera_tip_target(q_rad, displacement, rotation, solver=self.solver)
        if result is None:
            raise ValueError("현재 관절 범위에서 삽입 목표의 IK 해를 찾지 못했습니다.")
        target, scale = result
        # 축소된 실제 높이 계획량: $$s_{h,actual}=ks_h$$
        actual_height = scale * height
        # 축소된 실제 횡방향 계획량: $$s_{y,actual}=ks_y$$
        actual_lateral = scale * lateral
        return InsertionStep(target, actual_height, actual_lateral, fine and height_fine)
