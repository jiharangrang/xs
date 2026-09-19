"""관측한 빔 법선으로 카메라의 작은 정면 보정 자세를 계산한다.
공통 FK와 관절 제한을 사용하며 관측 오차의 고정값이나 모터 통신은 포함하지 않는다.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from kinematics.fk import ForwardKinematics
from kinematics.ik import joint_motion_cost, joint_motion_gradient
from kinematics.joints import as_joint_angles


def unit_normal(normal) -> np.ndarray:
    r"""카메라를 향하는 유한한 평면 법선을 단위 벡터로 반환한다.

    $$n=\tilde n/\|\tilde n\|$$
    """
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-8:
        raise ValueError("유효한 3차원 빔 법선이 필요합니다.")
    # 관측 법선의 길이 정규화: $$n=\tilde n/\|\tilde n\|$$
    normal = normal / np.linalg.norm(normal)
    if normal[2] >= 0:
        raise ValueError("법선은 카메라를 향해야 합니다.")
    return normal


def tilt_degrees(normal) -> float:
    r"""평면 법선과 카메라 정면 사이의 기울기를 도 단위로 계산한다.

    $$\theta=\cos^{-1}(-n_z)180/\pi$$
    """
    normal = unit_normal(normal)
    # 정면에서 벗어난 각도: $$\theta=\cos^{-1}(-n_z)180/\pi$$
    return float(np.rad2deg(np.arccos(np.clip(-normal[2], -1, 1))))


@dataclass(frozen=True)
class AlignmentSettings:
    """한 번의 보정 크기와 시작 자세 주변에서 허용할 관절 변화 범위를 지정한다."""

    gain: float = .6
    max_camera_step_deg: float = 2.
    max_joint_step_deg: float = 3.
    max_total_joint_deg: float = 20.
    max_observed_tilt_deg: float = 30.
    tip_tolerance_m: float = .0005

    def __post_init__(self):
        """음수·무한대 설정과 과도한 보정 이득을 거부한다."""
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()) or self.gain > 1:
            raise ValueError("보정 설정은 유한한 양수이고 이득은 1 이하여야 합니다.")


@dataclass(frozen=True)
class AlignmentStep:
    """작은 보정의 관절 목표와 모델상 예상 기울기·팁 위치 오차를 담는다."""

    q_rad: np.ndarray
    predicted_normal: np.ndarray
    predicted_tilt_deg: float
    tip_error_m: float


class BeamAlignmentPlanner:
    """L 고정 상태에서 R팁 위치를 유지하면서 깊이 카메라 방향만 조금 바꾼다."""

    def __init__(self, *, fk=None, settings=None):
        """기존 FK와 최적화 비용을 재사용할 작은 자세 솔버를 준비한다."""
        self.fk = fk if fk is not None else ForwardKinematics()
        self.settings = settings if settings is not None else AlignmentSettings()

    def plan(self, q_current, normal, q_reference) -> AlignmentStep:
        r"""현재 관측의 정면 회전을 제한하고 팁 위치 제약 아래 최소 관절 변화를 구한다.

        $$q^*=\arg\min_q\tfrac12\|q-q_c\|^2,
        \quad p_R(q)=p_R(q_0),\quad R_C(q)=R_C(q_c)\exp([\omega_C]_\times)$$

        q_c는 현재 실측각, q_0는 이번 보정 시작각이다. 관측 법선을 향해 광축을
        돌리는 최소 회전을 사용하므로 빔 길이 방향 위치와 광축 주위 회전은 목표로 삼지 않는다.
        """
        q_current, q_reference = as_joint_angles(q_current), as_joint_angles(q_reference)
        normal = unit_normal(normal)
        settings = self.settings
        tilt = tilt_degrees(normal)
        if tilt > settings.max_observed_tilt_deg:
            raise ValueError("정면 보정 범위를 벗어난 기울기입니다. 시작 자세와 빔 관측을 확인해 주세요.")
        limits = self.fk.joint_limits
        if np.any(q_current < limits[:, 0]) or np.any(q_current > limits[:, 1]):
            raise ValueError("현재 관절각이 공통 제한 범위를 벗어납니다.")
        tip_goal = self.fk.forward(q_reference).T_world_tip_R[:3, 3]
        camera = self.fk.depth_camera_pose(q_current)
        # 회전축은 현재 광축과 관측한 빔 방향의 외적: $$a=e_z\times(-n)$$
        axis = np.cross([0., 0., 1.], -normal)
        if np.linalg.norm(axis) < 1e-8:
            return AlignmentStep(q_current, normal, tilt, 0.)
        # 최소 회전의 단위축: $$\hat a=a/\|a\|$$
        axis /= np.linalg.norm(axis)
        # 보정각을 이득과 한 번의 회전 한도로 제한: $$\alpha=\min(k\theta,\alpha_{max})\pi/180$$
        angle = np.deg2rad(min(settings.gain * tilt, settings.max_camera_step_deg))
        # 한 번의 관절 변화 한도를 라디안으로 변환: $$s=s_{deg}\pi/180$$
        step_bound = np.deg2rad(settings.max_joint_step_deg)
        # 전체 관절 변화 한도를 라디안으로 변환: $$b=b_{deg}\pi/180$$
        total_bound = np.deg2rad(settings.max_total_joint_deg)
        lower = np.maximum.reduce([limits[:, 0], q_current - step_bound, q_reference - total_bound])
        upper = np.minimum.reduce([limits[:, 1], q_current + step_bound, q_reference + total_bound])
        if np.any(lower >= upper) or np.any(q_current < lower) or np.any(q_current > upper):
            raise ValueError("이번 보정의 관절 변화 범위를 벗어났습니다.")

        for scale in (1., .5, .25, .125):
            # 현재 카메라 기준의 작은 회전을 월드 방향에 합성: $$R_d=R_C\exp([s\alpha\hat a]_\times)$$
            rotation_goal = camera[:3, :3] @ Rotation.from_rotvec(scale * angle * axis).as_matrix()

            def constraint(q):
                r"""팁 위치와 카메라 방향의 등식 제약을 반환한다.

                $$c(q)=[(p_R(q)-p_d)/0.1;\operatorname{Log}(R_d^T R_C(q))^\vee]$$
                """
                tip = self.fk.forward(q).T_world_tip_R[:3, 3]
                orientation = self.fk.depth_camera_pose(q)[:3, :3]
                # 위치 오차의 수치 크기를 회전과 맞춤: $$c_p=(p_R-p_d)/0.1$$
                position_error = (tip - tip_goal) / .1
                # 목표 방향에서 본 회전 오차: $$c_R=\operatorname{Log}(R_d^TR_C)^\vee$$
                rotation_error = Rotation.from_matrix(rotation_goal.T @ orientation).as_rotvec()
                return np.concatenate([position_error, rotation_error])

            result = minimize(joint_motion_cost, q_current, args=(q_current,), jac=joint_motion_gradient,
                              method="SLSQP", bounds=list(zip(lower, upper)),
                              constraints={"type": "eq", "fun": constraint},
                              options={"maxiter": 100, "ftol": 1e-10})
            if not result.success or not np.all(np.isfinite(result.x)):
                continue
            error = constraint(result.x)
            # 정규화했던 위치 잔차를 미터로 복원: $$\varepsilon_p=0.1\|c_p\|$$
            tip_error = float(.1 * np.linalg.norm(error[:3]))
            if tip_error > settings.tip_tolerance_m or np.linalg.norm(error[3:]) > 1e-3:
                continue
            next_camera = self.fk.depth_camera_pose(result.x)
            # 동일한 평면 법선을 이동 후 카메라 기준으로 표현: $$n_{next}=R_{next}^TR_C n$$
            predicted = next_camera[:3, :3].T @ camera[:3, :3] @ normal
            predicted_tilt = tilt_degrees(predicted)
            if predicted_tilt < tilt - .05:
                return AlignmentStep(result.x.copy(), predicted, predicted_tilt, tip_error)
        raise ValueError("작은 보정 범위에서 팁 위치를 유지할 자세를 찾지 못했습니다.")
