"""관측 빔을 향한 수직 이동과 카메라 정면 회전을 관절 목표로 변환한다.
단계별 목표 판정과 분리해 공통 기구학·최적화 비용을 재사용한다.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from kinematics.anchoring import TipAnchor
from kinematics.ik import joint_motion_cost, joint_motion_gradient
from kinematics.joints import as_joint_angles
from planning.linear_motion import plan_linear_motion


def plan_vertical_step(q_rad, normal, distance_m, *, solver, steps, max_joint_step_deg, path_is_clear=None):
    r"""관절 변화와 선택적인 경로 조건을 만족하는 첫 상승 목표와 이동량을 반환한다.

    $$\Delta p_W=-ksR_{WC}n_C,\quad k\in\{1,1/2,1/4\}$$

    s는 요청 상승량이고 n_C는 카메라로 관측한 빔 법선이다.
    사용 가능한 후보가 없으면 단계별 중지 사유를 선택하도록 None을 반환한다.
    """
    start = solver.fk.forward(q_rad)
    camera = solver.fk.depth_camera_pose(q_rad)
    anchor = TipAnchor("tip_L", start.T_world_tip_L)
    for scale in (1., .5, .25):
        # 관측 빔 법선 반대 방향으로 작은 월드 상승: $$\Delta p_W=-ksR_{WC}n_C$$
        displacement = -scale * distance_m * (camera[:3, :3] @ normal)
        result = plan_linear_motion(q_rad, anchor, displacement, steps=steps, solver=solver)
        if not result.success:
            continue
        target = result.q_path_rad[-1]
        # 한 번 전송하는 관절 목표의 최대 변화: $$\Delta q_{max}=\max_j|q_{1,j}-q_{0,j}|180/\pi$$
        joint_change = float(np.max(np.abs(np.rad2deg(target - q_rad))))
        if joint_change > max_joint_step_deg:
            continue
        if path_is_clear is not None and not path_is_clear(target):
            continue
        # 축소 비율을 반영한 계획 상승량: $$s_{actual}=ks$$
        actual_distance = scale * distance_m
        return target.copy(), actual_distance
    return None


def solve_height_alignment_target(q_rad, normal, distance_m, rotation_vector, *, solver):
    r"""팁의 높이 변화와 카메라 정면 회전을 하나의 관절 목표로 함께 푼다.

    $$q^*=\arg\min_q\tfrac12\|q-q_c\|^2,\quad
    p_R(q)=p_R(q_c)-ksR_C(q_c)n,\quad
    R_C(q)=R_C(q_c)\exp([k\omega]_\times)$$

    s는 부호 있는 높이 보정량, omega는 현재 카메라 좌표의 정면 보정 회전이다.
    k를 줄이며 같은 목표 방향을 재시도하고 공통 관절 제한과 IK 잔차만 검사한다.
    """
    # 높이 이동을 카메라 기준 변위로 표현: $$\Delta p_C=-sn$$
    displacement = -distance_m * np.asarray(normal)
    result = solve_camera_tip_target(q_rad, displacement, rotation_vector, solver=solver)
    if result is None:
        return None
    target, scale = result
    # 축소 비율을 반영한 실제 높이 이동량: $$s_{actual}=ks$$
    actual_distance = scale * distance_m
    return target, actual_distance


def solve_camera_tip_target(q_rad, displacement_camera_m, rotation_vector, *, solver):
    r"""카메라 좌표의 팁 변위와 카메라 회전을 하나의 관절 목표로 푼다.

    $$q^*=\arg\min_q\tfrac12\|q-q_c\|^2,\quad
    p_R(q)=p_R(q_c)+kR_C(q_c)\Delta p_C,\quad
    R_C(q)=R_C(q_c)\exp([k\omega]_\times)$$

    변위와 회전을 같은 비율 k로 줄여 재시도하고 성공한 목표와 적용 비율을 반환한다.
    """
    q_rad = as_joint_angles(q_rad)
    displacement_camera_m = np.asarray(displacement_camera_m, dtype=float)
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (displacement_camera_m, rotation_vector)):
        raise ValueError("이동과 회전 목표는 유한한 3차원 벡터여야 합니다.")
    limits, settings = solver.fk.joint_limits, solver.settings
    if np.any(q_rad < limits[:, 0]) or np.any(q_rad > limits[:, 1]):
        raise ValueError("현재 관절각이 공통 제한 범위를 벗어납니다.")
    tip = solver.fk.forward(q_rad).T_world_tip_R[:3, 3].copy()
    camera = solver.fk.depth_camera_pose(q_rad)
    for scale in (1., .5, .25, .125):
        # 관측 좌표의 변위를 월드 팁 목표에 반영: $$p_d=p_R+kR_C\Delta p_C$$
        position_goal = tip + scale * (camera[:3, :3] @ displacement_camera_m)
        # 같은 보정에서 카메라 방향 목표도 설정: $$R_d=R_C\exp([k\omega]_\times)$$
        rotation_goal = camera[:3, :3] @ Rotation.from_rotvec(scale * rotation_vector).as_matrix()

        def constraint(q):
            r"""이동할 팁 위치와 카메라 방향의 잔차를 하나의 등식 제약으로 반환한다.

            $$c(q)=[(p_R(q)-p_d)/\ell;\operatorname{Log}(R_d^TR_C(q))^\vee]$$
            """
            current_tip = solver.fk.forward(q).T_world_tip_R[:3, 3]
            orientation = solver.fk.depth_camera_pose(q)[:3, :3]
            # 길이 척도로 정규화한 위치 잔차: $$c_p=(p_R-p_d)/\ell$$
            position_error = (current_tip - position_goal) / settings.position_scale_m
            # 목표 방향에서 본 카메라 회전 잔차: $$c_R=\operatorname{Log}(R_d^TR_C)^\vee$$
            rotation_error = Rotation.from_matrix(rotation_goal.T @ orientation).as_rotvec()
            return np.concatenate([position_error, rotation_error])

        result = minimize(joint_motion_cost, q_rad, args=(q_rad,), jac=joint_motion_gradient,
                          method="SLSQP", bounds=list(map(tuple, limits)),
                          constraints={"type": "eq", "fun": constraint},
                          options={"maxiter": settings.max_iterations, "ftol": settings.optimizer_tolerance})
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        if np.any(result.x < limits[:, 0]) or np.any(result.x > limits[:, 1]):
            continue
        error = constraint(result.x)
        # 정규화한 잔차를 미터로 복원: $$\varepsilon_p=\ell\|c_p\|$$
        position_error = settings.position_scale_m * np.linalg.norm(error[:3])
        if position_error > settings.position_tolerance_m or np.linalg.norm(error[3:]) > settings.rotation_tolerance_rad:
            continue
        return result.x.copy(), scale
    return None
