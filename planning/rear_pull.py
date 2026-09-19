"""현재 자세와 관측한 빔 방향으로 앞 팁을 고정한 뒷 팁 당김 경로를 만든다.
개방·고정각 외측 회전·당김·출발 횡위치 복귀를 구성하며 실물 처짐 보정은 수행하지 않는다.
"""

from dataclasses import dataclass

import numpy as np

from kinematics.anchoring import TipAnchor
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles
from planning.linear_motion import plan_linear_motion
from planning.motion_path import MotionPath, MotionSegment


REAR_OPEN_DEG = -120.
DEFAULT_PULL_MM = 100.
# 뒷그리퍼 옆 빼기에 사용할 고정 회전량이다.
REAR_SIDE_YAW_DEG = 3.


@dataclass(frozen=True)
class RearReturnReference:
    """옆으로 빠지기 전의 뒷 팁 위치와 빔 횡방향을 같은 고정점 좌표계에 보관한다."""

    point_world_m: np.ndarray
    outward_world: np.ndarray

    def __post_init__(self):
        r"""복귀 기준의 형상과 유효성을 확인하고 횡방향을 정규화한다.

        $$u_W=\tilde u_W/\|\tilde u_W\|$$
        """
        point = np.array(self.point_world_m, dtype=float, copy=True)
        outward = np.array(self.outward_world, dtype=float, copy=True)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (point, outward)):
            raise ValueError("뒷그리퍼 복귀 위치와 횡방향이 유효하지 않습니다.")
        if np.linalg.norm(outward) < 1e-8:
            raise ValueError("뒷그리퍼 복귀 횡방향을 구할 수 없습니다.")
        # 빔 횡방향의 단위 벡터: $$u_W=\tilde u_W/\|\tilde u_W\|$$
        outward /= np.linalg.norm(outward)
        point.setflags(write=False)
        outward.setflags(write=False)
        object.__setattr__(self, "point_world_m", point)
        object.__setattr__(self, "outward_world", outward)

    def as_dict(self):
        """현재 실행의 상태와 로그에 기록할 복귀 기준을 반환한다."""
        return {"kind": "stage7_start_lateral", "point_world_m": self.point_world_m.tolist(),
                "outward_world": self.outward_world.tolist()}


@dataclass
class RearPullPlan:
    """앞 팁 고정 경로와 관측에서 정한 전진 방향을 함께 보관한다."""

    motion: MotionPath
    direction_world: np.ndarray
    distance_m: float
    max_position_error_m: float
    max_rotation_error_rad: float
    return_reference: RearReturnReference

    @property
    def pull(self):
        """실제 몸통 이동에 사용할 당김 구간을 반환한다."""
        return next(segment for segment in self.motion.segments if segment.name == "rear_pull")

    @property
    def side_exit(self):
        """고정각 외측 회전 구간을 반환하며 이미 빠진 뒤의 재계획에서는 생략한다."""
        return next((segment for segment in self.motion.segments if segment.name == "rear_side_exit"), None)

    @property
    def side_return(self):
        """당김 이후 출발 횡위치로 복귀하는 구간을 반환한다."""
        return next(segment for segment in self.motion.segments if segment.name == "rear_side_return")


class RearPullPlanner:
    """기존 직선 IK를 현재 실측 자세와 오른쪽 지지점에 연결한다."""

    def __init__(self, solver=None):
        """실물 장치에 접근하지 않고 재사용할 IK만 준비한다."""
        self.solver = solver if solver is not None else InverseKinematics(settings=IKSettings(starts=1))

    def plan(self, q_rad, grippers_deg, axis_camera, distance_mm=DEFAULT_PULL_MM, *,
             include_side_exit=True, anchor=None, return_reference=None):
        r"""고정각으로 옆으로 빠진 자세에서 관측 빔을 따라 앞 팁 쪽으로 이동시킨다.

        $$q'_1=q_1+\theta,\quad q'_7=q_7-\theta,\quad
        \Delta p_W=d\,\operatorname{sgn}(a_W^T(p_R-p_L))a_W$$

        theta는 REAR_SIDE_YAW_DEG의 라디안 값이다. 다른 몸통 관절은 회전 구간에서 유지한다.
        외측 회전은 관절 목표만 정하며 실제 옆 여유나 일정 높이를 보장하지 않는다.
        길이 방향 진행량은 기구학으로 정하며 카메라로 실측한 이동량이 아니다.
        """
        if not np.isfinite(distance_mm) or not 0 < distance_mm <= 100:
            raise ValueError("뒷그리퍼 이동량은 0 초과 100 mm 이하로 입력해 주세요.")
        axis = np.asarray(axis_camera, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-8:
            raise ValueError("관측한 빔 길이 방향이 유효하지 않습니다.")
        q_rad = as_joint_angles(q_rad)
        fk = self.solver.fk
        model_start = fk.forward(q_rad)
        anchor = anchor if anchor is not None else TipAnchor("tip_R", model_start.T_world_tip_R)
        if anchor.fixed_tip != "tip_R":
            raise ValueError("뒷그리퍼 당김은 앞 팁을 고정점으로 사용합니다.")
        start = anchor.place(model_start)
        # 재계획에서도 같은 앞 팁 고정 좌표계의 카메라 자세를 사용: $$T_{WC}=T_{WM}T_{MC}$$
        camera = anchor.world_from_model(model_start) @ fk.depth_camera_pose(q_rad)
        # 관측 방향을 모델 월드의 단위 방향으로 변환: $$a_W=R_{WC}a_C/\|a_C\|$$
        direction = camera[:3, :3] @ axis / np.linalg.norm(axis)
        # 앞뒤 팁을 연결하는 방향: $$v=p_R-p_L$$
        separation = start.T_world_tip_R[:3, 3] - start.T_world_tip_L[:3, 3]
        if abs(direction @ separation) < 1e-6:
            raise ValueError("빔 방향에서 앞뒤 그리퍼 순서를 구분하지 못했습니다.")
        if direction @ separation < 0:
            direction = -direction
        # 입력한 밀리미터 거리를 미터로 변환: $$d=d_{mm}/1000$$
        distance_m = distance_mm / 1000
        # 빔 방향의 당김 변위: $$\Delta p_W=d a_W$$
        displacement = distance_m * direction
        # 직선을 최대 2 mm 간격으로 나누기: $$K=\lceil d/0.002\rceil$$
        steps = max(1, int(np.ceil(distance_m / .002)))
        q_side = q_rad.copy()
        if include_side_exit:
            # 뒷 그리퍼의 방향 변화를 줄이는 반대쪽 회전: $$q'_1=q_1+\theta$$
            q_side[0] += np.deg2rad(REAR_SIDE_YAW_DEG)
            # 앞 지지점을 중심으로 뒷 그리퍼를 바깥으로 회전: $$q'_7=q_7-\theta$$
            q_side[6] -= np.deg2rad(REAR_SIDE_YAW_DEG)
            side = anchor.place(fk.forward(q_side))
            # 빔 길이와 현재 뒷 팁 높이축에 수직인 횡방향: $$\tilde u_W=a_W\times z_{L,W}$$
            outward = np.cross(direction, start.T_world_tip_L[:3, 2])
            # 외측 회전으로 생기는 팁 변위: $$\Delta p_s=p_{L,side}-p_{L,start}$$
            side_displacement = side.T_world_tip_L[:3, 3] - start.T_world_tip_L[:3, 3]
            if outward @ side_displacement < 0:
                outward = -outward
            return_reference = RearReturnReference(start.T_world_tip_L[:3, 3], outward)
        elif return_reference is None:
            raise ValueError("옆 빼기 후에는 빠지기 전의 뒷그리퍼 복귀 기준이 필요합니다.")
        result = plan_linear_motion(q_side, anchor, displacement, steps=steps, solver=self.solver)
        if not result.success:
            raise ValueError(result.message)
        # 현재 두 손가락 각도를 라디안으로 변환: $$g_0=g_{deg}\pi/180$$
        grips_start = np.deg2rad([grippers_deg["G_L"], grippers_deg["G_R"]])
        grips_open = grips_start.copy()
        # 뒷 손가락만 개방 목표로 변경: $$g_L=-120\pi/180$$
        grips_open[0] = np.deg2rad(REAR_OPEN_DEG)
        # 화면 재생용 개방 시간: $$T_o=\max(0.5,|g_L-g_{L,0}|/v_g)$$
        opening_s = max(.5, abs(REAR_OPEN_DEG - grippers_deg["G_L"]) / 10.)
        # 관절 속도를 기준으로 정한 경로 시간이며 실물 추종은 별도 확인: $$\Delta t_k=\max(0.2,\max_j|\Delta q_{k,j}|/v_q)$$
        durations = np.maximum(.2, np.max(np.abs(np.rad2deg(np.diff(result.q_path_rad, axis=0))), axis=1) / 3.)
        segments = [MotionSegment("rear_open", anchor, [0., opening_s], [q_rad, q_rad], [grips_start, grips_open])]
        if include_side_exit:
            # 화면 재생용 고정각 회전 시간: $$T_s=\theta_{deg}/3$$
            side_duration = REAR_SIDE_YAW_DEG / 3.
            segments.append(MotionSegment("rear_side_exit", anchor, [0., side_duration], [q_rad, q_side],
                                          [grips_open, grips_open]))
        segments.append(MotionSegment("rear_pull", anchor, np.r_[0., np.cumsum(durations)], result.q_path_rad,
                                      np.tile(grips_open, (steps + 1, 1))))
        segments.append(self.plan_side_return(result.q_path_rad[-1], {**grippers_deg, "G_L": REAR_OPEN_DEG},
                                             anchor, return_reference))
        motion = MotionPath(tuple(segments))
        motion.validate_limits()
        return RearPullPlan(motion, direction, distance_m, result.max_position_error_m,
                            result.max_rotation_error_rad, return_reference)

    def plan_side_return(self, q_rad, grippers_deg, anchor, reference):
        r"""당김 도착 자세에서 길이·높이·방향을 유지하며 출발 횡위치로 돌아간다.

        $$\Delta p_W=-[u_W^T(p_L-p_0)]u_W$$

        옆 빼기의 역회전 대신 현재 링크 형상에서 필요한 횡변위를 IK로 계산한다.
        p_0와 u_W는 옆 빼기 전의 모델 기준이며 실물 접촉이나 파지를 판정하지 않는다.
        """
        if anchor.fixed_tip != "tip_R":
            raise ValueError("뒷그리퍼 복귀는 앞 팁을 고정점으로 사용합니다.")
        q_rad = as_joint_angles(q_rad)
        current = anchor.place(self.solver.fk.forward(q_rad)).T_world_tip_L
        # 출발 횡위치에서 벗어난 거리: $$e_y=u_W^T(p_L-p_0)$$
        remaining = float(reference.outward_world @ (current[:3, 3] - reference.point_world_m))
        # 길이·높이 성분을 제외한 횡복귀 변위: $$\Delta p_W=-e_yu_W$$
        displacement = -remaining * reference.outward_world
        # 복귀 직선을 최대 2 mm 간격으로 분할: $$K=\max(1,\lceil |e_y|/0.002\rceil)$$
        steps = max(1, int(np.ceil(abs(remaining) / .002)))
        result = plan_linear_motion(q_rad, anchor, displacement, steps=steps, solver=self.solver)
        if not result.success:
            raise ValueError(f"뒷그리퍼 횡복귀 경로: {result.message}")
        # 몸통 속도에 맞춘 복귀 경로 시간: $$\Delta t_k=\max(0.2,\max_j|\Delta q_{k,j}|/3)$$
        durations = np.maximum(.2, np.max(np.abs(np.rad2deg(np.diff(result.q_path_rad, axis=0))), axis=1) / 3.)
        # 두 그리퍼의 현재각을 라디안으로 변환: $$g=g_{deg}\pi/180$$
        grips = np.deg2rad([grippers_deg["G_L"], grippers_deg["G_R"]])
        segment = MotionSegment("rear_side_return", anchor, np.r_[0., np.cumsum(durations)], result.q_path_rad,
                                np.tile(grips, (steps + 1, 1)))
        MotionPath((segment,)).validate_limits()
        return segment
