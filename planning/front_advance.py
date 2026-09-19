"""뒷발을 고정한 앞발 전진에서 빔 높이·정면·횡위치를 함께 보정한다.
길이·횡방향은 관절 모델로 추정하고 높이·정면은 새 깊이 관측을 사용한다.
"""

from dataclasses import dataclass

import numpy as np

from kinematics.anchoring import TipAnchor
from kinematics.joints import as_joint_angles
from planning.beam_alignment import tilt_degrees, unit_normal
from planning.insertion_height import InsertionHeightPlanner
from planning.linear_motion import plan_linear_motion
from planning.motion_path import MotionPath, MotionSegment
from planning.vertical_motion import solve_camera_tip_target


FRONT_OPEN_DEG = -120.
FRONT_CLOSE_DEG = 4.6
DEFAULT_ADVANCE_MM = 100.


@dataclass(frozen=True)
class FrontAdvanceSettings:
    """직진의 길이·횡오차와 완료 높이 허용오차 및 한 번의 이동량을 지정한다."""

    tolerance_m: float = .001
    arrival_height_tolerance_m: float = .002
    max_step_m: float = .005

    def __post_init__(self):
        """거리 설정이 유한한 양수인지 확인한다."""
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("앞발 전진 거리 설정은 유한한 양수여야 합니다.")


@dataclass(frozen=True)
class FrontAdvanceReference:
    """현재 실행의 출발 위치와 전진·횡방향 및 전진 목표 거리를 보관한다."""

    point_world_m: np.ndarray
    axis_world: np.ndarray
    outward_world: np.ndarray
    distance_m: float

    def as_dict(self):
        """현재 실행에서만 사용하는 전진 기준을 상태와 로그에 전달한다."""
        return {"point_world_m": self.point_world_m.tolist(), "axis_world": self.axis_world.tolist(),
                "outward_world": self.outward_world.tolist(), "distance_m": self.distance_m}


@dataclass(frozen=True)
class FrontAdvanceMeasurement:
    """모델 기준 길이·횡오차와 카메라 기준 높이·옆 여유를 구분해 담는다."""

    progress_m: float
    forward_error_m: float
    lateral_error_m: float
    height_error_m: float
    side_clearance_m: float
    depth_m: float
    goal_depth_m: float
    tilt_deg: float
    normal: np.ndarray
    outward: np.ndarray


@dataclass(frozen=True)
class FrontAdvanceStep:
    """전진과 횡이동·높이·정면 보정이 함께 적용된 관절 목표를 담는다."""

    q_rad: np.ndarray
    distance_m: float
    forward_m: float
    lateral_m: float
    fine: bool


class FrontAdvancePlanner:
    """앞발 카메라의 높이 보정과 기존 팁 이동 IK를 결합한다."""

    def __init__(self, *, settings=None, height_settings=None):
        """모터에 접근하지 않고 공통 높이·형상·IK 모듈을 준비한다."""
        self.settings = settings or FrontAdvanceSettings()
        self.height = InsertionHeightPlanner(settings=height_settings)
        self.geometry, self.solver = self.height.geometry, self.height.solver
        self.reference = None

    def outward_hint(self, q_rad, grippers_deg):
        """앞 고정턱 쪽의 모서리를 선택할 관측 방향을 반환한다."""
        return self.height.outward_hint(q_rad, grippers_deg)

    def configure(self, q_rad, grippers_deg, observation, distance_mm=DEFAULT_ADVANCE_MM):
        r"""현재 파지 위치와 새 빔 방향에서 이번 전진의 독립 기준을 만든다.

        $$a_W=R_{WC}a_C,\quad u_W=R_{WC}u_C,\quad p_0=p_R(q_0)$$
        """
        if not np.isfinite(distance_mm) or not 0 < distance_mm <= 100:
            raise ValueError("앞발 전진량은 0 초과 100 mm 이하로 입력해 주세요.")
        q_rad = as_joint_angles(q_rad)
        normal = unit_normal(observation.normal)
        axis = np.asarray(observation.axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("빔의 전진 방향이 유효하지 않습니다.")
        # 빔 평면 안의 길이 방향: $$\tilde a_C=a_C-(a_C^Tn_C)n_C$$
        axis = axis - float(axis @ normal) * normal
        if np.linalg.norm(axis) < 1e-8:
            raise ValueError("빔 평면에서 전진 방향을 구할 수 없습니다.")
        # 길이 방향 정규화: $$a_C=\tilde a_C/\|\tilde a_C\|$$
        axis /= np.linalg.norm(axis)
        camera = self.solver.fk.depth_camera_pose(q_rad)
        start = self.solver.fk.forward(q_rad)
        # 관측 길이 방향을 뒷발 고정 모델 좌표로 변환: $$a_W=R_{WC}a_C$$
        direction = camera[:3, :3] @ axis
        # 뒷발에서 앞발을 향하는 방향: $$v=p_R-p_L$$
        separation = start.T_world_tip_R[:3, 3] - start.T_world_tip_L[:3, 3]
        if abs(direction @ separation) < 1e-6:
            raise ValueError("빔 방향에서 앞발의 전진 방향을 정하지 못했습니다.")
        if direction @ separation < 0:
            direction = -direction
        # 길이와 높이에 수직인 카메라 횡방향: $$u_C=n_C\times a_C$$
        outward = np.cross(normal, axis)
        hint = self.outward_hint(q_rad, grippers_deg)
        if outward @ hint < 0:
            outward = -outward
        # 고정턱 쪽 횡방향을 뒷발 고정 모델 좌표로 변환: $$u_W=R_{WC}u_C$$
        outward_world = camera[:3, :3] @ outward
        # 전진 목표의 미터 단위 변환: $$d=d_{mm}/1000$$
        distance_m = distance_mm / 1000
        self.reference = FrontAdvanceReference(start.T_world_tip_R[:3, 3].copy(), direction,
                                                outward_world, distance_m)
        return self.reference

    def measure(self, q_rad, grippers_deg, observation):
        r"""현재 전진량·횡오차와 관측 높이·옆 여유를 계산한다.

        $$s=a_W^T(p_R-p_0),\quad e_y=u_W^T(p_R-p_0),\quad
        g_y=\min_i u_C^T(p_{lip,i}-p_e)$$
        """
        if self.reference is None:
            raise ValueError("앞발 전진의 출발 기준이 필요합니다.")
        normal = unit_normal(observation.normal)
        outward = np.asarray(observation.outward, dtype=float)
        point = np.asarray(observation.edge_point_m, dtype=float)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (outward, point)):
            raise ValueError("빔 모서리 관측이 유효하지 않습니다.")
        # 높이 성분을 제거한 횡방향: $$\tilde u_C=u_C-(u_C^Tn_C)n_C$$
        outward = outward - float(outward @ normal) * normal
        if np.linalg.norm(outward) < 1e-8:
            raise ValueError("빔의 횡방향이 유효하지 않습니다.")
        # 카메라 횡방향 정규화: $$u_C=\tilde u_C/\|\tilde u_C\|$$
        outward /= np.linalg.norm(outward)
        self.geometry.update(q_rad, grippers_deg)
        if outward @ self.geometry.outward_camera() <= 0:
            raise ValueError("앞 고정턱 쪽의 빔 모서리가 필요합니다.")
        lip = self.geometry.points_camera(fixed_lip=True)
        # 앞 고정턱 안쪽 면의 옆 여유: $$g_y=\min_i u_C^T(p_{lip,i}-p_e)$$
        clearance = float(np.min((lip - point) @ outward))
        height_error, goal_depth = self.height.height_error(q_rad, normal, observation.plane_offset_m)
        tip = self.solver.fk.forward(q_rad).T_world_tip_R[:3, 3]
        # 출발 위치에서의 모델 변위: $$\Delta p=p_R-p_0$$
        displacement = tip - self.reference.point_world_m
        # 길이 방향의 모델 진행량: $$s=a_W^T\Delta p$$
        progress = float(self.reference.axis_world @ displacement)
        # 목표 길이까지 남은 거리: $$e_x=d-s$$
        forward_error = self.reference.distance_m - progress
        # 출발 횡위치까지의 복귀 거리: $$e_y=u_W^T\Delta p$$
        lateral_error = float(self.reference.outward_world @ displacement)
        return FrontAdvanceMeasurement(progress, forward_error, lateral_error, height_error, clearance,
                                        float(observation.plane_offset_m), goal_depth, tilt_degrees(normal), normal, outward)

    def plan(self, q_rad, grippers_deg, observation):
        r"""전진·횡이동·높이 보정과 정면 회전을 하나의 관절 목표로 푼다.

        $$\Delta p_C=s_xa_C+s_yu_C-s_hn_C$$
        """
        measured = self.measure(q_rad, grippers_deg, observation)
        settings = self.settings
        forward = 0.
        if abs(measured.forward_error_m) > settings.tolerance_m:
            # 한 번에 진행할 길이 방향 거리: $$s_x=\operatorname{clip}(e_x,-s_{max},s_{max})$$
            forward = float(np.clip(measured.forward_error_m, -settings.max_step_m, settings.max_step_m))
        # 옆으로 빠지지 않고 출발 직선의 횡위치를 유지: $$e_s=-e_y$$
        side_error = -measured.lateral_error_m
        # 허용 범위를 벗어난 횡오차만 반영: $$s_y=\operatorname{clip}(e_s,-s_{max},s_{max})$$
        lateral = 0. if abs(side_error) <= settings.tolerance_m else float(np.clip(side_error, -settings.max_step_m, settings.max_step_m))
        height, rotation, _ = self.height.correction(measured.height_error_m, measured.normal)
        camera = self.solver.fk.depth_camera_pose(q_rad)
        # 같은 전진 기준을 현재 카메라 좌표로 변환: $$a_C=R_{WC}^Ta_W$$
        axis = camera[:3, :3].T @ self.reference.axis_world
        # 같은 직진 경로의 횡방향을 현재 카메라 좌표로 변환: $$u_C=R_{WC}^Tu_W$$
        outward = camera[:3, :3].T @ self.reference.outward_world
        # 전진과 높이·횡방향 이동을 같은 목표에 합성: $$\Delta p_C=s_xa_C+s_yu_C-s_hn_C$$
        displacement = forward * axis + lateral * outward - height * measured.normal
        result = solve_camera_tip_target(q_rad, displacement, rotation, solver=self.solver)
        if result is None:
            raise ValueError("현재 관절 범위에서 앞발 전진·높이·정면 보정 목표를 구하지 못했습니다.")
        target, scale = result
        fine = abs(side_error) < .004 and abs(measured.forward_error_m) < .004
        # IK 축소 비율을 적용한 높이 보정량: $$s_{h,actual}=ks_h$$
        actual_height = scale * height
        # IK 축소 비율을 적용한 길이 이동량: $$s_{x,actual}=ks_x$$
        actual_forward = scale * forward
        # IK 축소 비율을 적용한 횡이동량: $$s_{y,actual}=ks_y$$
        actual_lateral = scale * lateral
        return FrontAdvanceStep(target, actual_height, actual_forward, actual_lateral, fine)

    def preview(self, q_rad, grippers_deg, observation):
        r"""개방 전에 횡이탈 없는 직진·잠금의 모델 경로를 검사한다.

        $$\Delta p_{forward}=e_x a_W-e_yu_W$$

        실제 실행은 이 경로를 그대로 재생하지 않고 매 관측에서 높이·정면을 다시 보정한다.
        """
        measured = self.measure(q_rad, grippers_deg, observation)
        start = self.solver.fk.forward(q_rad)
        anchor = TipAnchor("tip_L", start.T_world_tip_L)
        # 그리퍼 표시각을 라디안으로 변환: $$g=g_{deg}\pi/180$$
        grips = np.deg2rad([grippers_deg["G_L"], grippers_deg["G_R"]])
        opened = grips.copy()
        # 앞 손가락만 개방: $$g_R=-120\pi/180$$
        opened[1] = np.deg2rad(FRONT_OPEN_DEG)
        segments = [MotionSegment("front_open", anchor, [0., 1.], [q_rad, q_rad], [grips, opened])]
        # 개방 전 출발 직선을 기준으로 남은 전진과 횡오차를 함께 반영: $$\Delta p_{forward}=e_x a_W-e_yu_W$$
        forward = measured.forward_error_m * self.reference.axis_world - measured.lateral_error_m * self.reference.outward_world
        segments.append(self._linear_segment("front_advance", q_rad, opened, anchor, forward))
        closed = opened.copy()
        # 앞 손가락의 잠금 목표: $$g_R=4.6\pi/180$$
        closed[1] = np.deg2rad(FRONT_CLOSE_DEG)
        q_end = segments[-1].q_rad[-1]
        segments.append(MotionSegment("front_close", anchor, [0., 1.], [q_end, q_end], [opened, closed]))
        path = MotionPath(tuple(segments))
        path.validate_limits()
        return path

    def _linear_segment(self, name, q_rad, grips, anchor, displacement):
        r"""보간 오차가 큰 경우 검사 간격을 줄여 사전 확인용 직선 경로를 만든다.

        $$K_r=2^r\max(1,\lceil\|\Delta p\|/0.002\rceil),\quad r\in\{0,1,2,3\}$$

        위치·방향 허용오차와 관절 한도는 유지하며 IK 실패는 세분화로 우회하지 않는다.
        """
        # 최대 2 mm 간격의 경로 분할 수: $$K=\max(1,\lceil\|\Delta p\|/0.002\rceil)$$
        steps = max(1, int(np.ceil(np.linalg.norm(displacement) / .002)))
        for refinement in range(4):
            result = plan_linear_motion(q_rad, anchor, displacement, steps=steps, solver=self.solver)
            if result.success:
                break
            sampling_error = (result.max_position_error_m > self.solver.settings.position_tolerance_m
                              or result.max_rotation_error_rad > self.solver.settings.rotation_tolerance_rad)
            if not sampling_error or refinement == 3:
                raise ValueError(f"앞발 경로 {name}: {result.message}")
            # 같은 경로의 검사 간격을 절반으로 축소: $$K_{r+1}=2K_r$$
            steps *= 2
        # 관절 속도를 기준으로 한 미리보기 시간: $$\Delta t_k=\max(0.2,\max_j|\Delta q_{k,j}|/3)$$
        durations = np.maximum(.2, np.max(np.abs(np.rad2deg(np.diff(result.q_path_rad, axis=0))), axis=1) / 3.)
        return MotionSegment(name, anchor, np.r_[0., np.cumsum(durations)], result.q_path_rad,
                             np.tile(grips, (steps + 1, 1)))
