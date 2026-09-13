"""팁 목표 자세와 관절 제한을 만족하면서 관절각 변화의 제곱합을 줄이는 IK를 푼다.
여러 시작값에서 SLSQP를 실행하고 중복을 제외한 상위 후보를 제한된 개수만 유지한다.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from kinematics.fk import DEFAULT_MODEL_PATH, ForwardKinematics
from kinematics.joints import as_joint_angles
from kinematics.poses import as_pose, pose_error


@dataclass(frozen=True)
class IKSettings:
    """시도 횟수, 반복 한도와 수치적인 도착 판정 기준을 지정한다.

    position_scale_m은 제약식의 수치 크기를 맞추는 길이 척도이다.
    위치·방향 허용오차는 최종 검증에 각각 적용하며 실물 정확도를 뜻하지 않는다.
    max_candidates는 보관 개수의 상한이며 duplicate_tolerance_rad는 후보를 묶는 각도 차이다.
    """

    starts: int = 24
    max_iterations: int = 300
    random_seed: int = 0
    max_candidates: int = 5
    duplicate_tolerance_rad: float = 1e-3
    position_tolerance_m: float = 1e-4
    rotation_tolerance_rad: float = 1e-3
    position_scale_m: float = 0.1
    optimizer_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """반복 설정과 오차 기준이 유효한지 검사한다."""
        for name in ("starts", "max_iterations", "max_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}는 양의 정수여야 합니다.")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed는 음이 아닌 정수여야 합니다.")
        for name in (
            "position_tolerance_m", "rotation_tolerance_rad",
            "position_scale_m", "optimizer_tolerance", "duplicate_tolerance_rad",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}는 유한한 양수여야 합니다.")


@dataclass
class IKCandidate:
    """목표와 관절 제한을 통과한 후보 하나의 관절각·비용·오차를 담는다.

    q_rad는 J1부터 J7 순서의 라디안 배열이다.
    position_error_m은 위치 오차의 길이, rotation_error_rad는 방향 차이의 각도다.
    cost는 현재 관절각 대비 변화 제곱합의 절반이다.
    """

    q_rad: NDArray[np.float64]
    cost: float
    position_error_m: float
    rotation_error_rad: float


@dataclass
class IKResult:
    """최선의 해와 비용 순으로 정렬한 서로 다른 상위 후보 및 계산 상태를 담는다.

    q_rad·cost·오차는 첫 번째 후보의 값이며 실패하면 None이다.
    candidates는 최대 max_candidates개의 후보이며 실패하면 빈 튜플이다.
    attempts는 실제 시도 횟수, solved_attempts는 수렴과 검증을 모두 통과한 횟수다.
    성공한 시도들이 같은 관절각으로 수렴하면 후보는 하나만 남는다.
    성공은 기구학적 해를 찾았다는 의미이며 충돌 검사와 전역 최적성은 포함하지 않는다.
    """

    success: bool
    q_rad: NDArray[np.float64] | None
    cost: float | None
    position_error_m: float | None
    rotation_error_rad: float | None
    attempts: int
    solved_attempts: int
    message: str
    candidates: tuple[IKCandidate, ...] = ()


def joint_motion_cost(
    q_rad: NDArray[np.float64], q_start_rad: NDArray[np.float64]
) -> float:
    r"""모든 관절에 같은 가중치를 적용한 관절각 변화 비용을 계산한다.

    $$
    C(q)=\tfrac12\|q-q_s\|_2^2
    $$

    q는 후보 관절각, q_s는 실제 시작 관절각이며 두 입력의 단위는 rad이다.
    """
    # 실제 시작 자세에서 후보 자세까지의 각도 차이: $$\Delta q=q-q_s$$
    displacement = q_rad - q_start_rad
    # 관절별 변화 제곱합의 절반: $$C=\tfrac12\Delta q^T\Delta q$$
    cost = 0.5 * np.dot(displacement, displacement)
    return float(cost)


def joint_motion_gradient(
    q_rad: NDArray[np.float64], q_start_rad: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""관절각 변화 비용을 각 관절각으로 미분한 기울기를 계산한다.

    $$
    \nabla_q C=q-q_s
    $$

    q는 후보 관절각, q_s는 비용의 기준이 되는 실제 시작 관절각이다.
    """
    # 비용의 관절각별 변화율: $$g=q-q_s$$
    gradient = q_rad - q_start_rad
    return gradient


class InverseKinematics:
    """FK 모델과 XML 관절 제한을 한 번 읽어 여러 상대 목표 자세의 IK를 계산한다."""

    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL_PATH, settings: IKSettings | None = None
    ) -> None:
        """FK와 설정을 준비하고 다중 시작값에 사용할 유한한 관절 범위를 읽는다."""
        self.fk = ForwardKinematics(model_path)
        self.settings = settings if settings is not None else IKSettings()
        self._limits = self.fk.joint_limits
        if not np.all(np.isfinite(self._limits)) or np.any(
            self._limits[:, 0] >= self._limits[:, 1]
        ):
            raise ValueError("IK에는 각 팔 관절의 유한하고 유효한 XML range가 필요합니다.")

    def _pose_constraint(
        self, q_rad: NDArray[np.float64], target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""위치와 방향이 목표와 일치할 때 영이 되는 등식 제약을 계산한다.

        $$
        c(q)=\begin{bmatrix}(p(q)-p_d)/\ell\\\operatorname{Log}(R_d^T R(q))^\vee\end{bmatrix}=0
        $$

        p와 R은 L팁에서 본 R팁의 자세이고 첨자 d는 target이다.
        ell은 position_scale_m이며 위치와 회전의 수치 크기를 맞춘다.
        """
        actual = self.fk.forward(q_rad).T_tip_L_tip_R
        error = pose_error(actual, target)
        # 위치 오차의 단위 크기 정규화: $$c_p=e_p/\ell$$
        error[:3] = error[:3] / self.settings.position_scale_m
        return error

    def _validated_candidate(
        self, q_rad: NDArray[np.float64], target: NDArray[np.float64],
        q_start: NDArray[np.float64],
    ) -> IKCandidate | None:
        r"""후보의 관절 범위와 위치·방향 오차를 검사해 통과한 결과만 반환한다.

        $$
        (\varepsilon_p,\varepsilon_R)=(\|e_p\|_2,\|e_R\|_2)
        $$

        e는 FK 자세와 target 사이의 오차이며, 각각 m와 rad로 판정한다.
        q_start는 반환 비용을 계산할 실제 시작 관절각이다.
        """
        if not np.all(np.isfinite(q_rad)):
            return None
        if np.any(q_rad < self._limits[:, 0]) or np.any(q_rad > self._limits[:, 1]):
            return None
        actual = self.fk.forward(q_rad).T_tip_L_tip_R
        error = pose_error(actual, target)
        # 위치 오차의 실제 길이: $$\varepsilon_p=\|e_p\|_2$$
        position_error = float(np.linalg.norm(error[:3]))
        # 목표 방향과의 회전각: $$\varepsilon_R=\|e_R\|_2$$
        rotation_error = float(np.linalg.norm(error[3:]))
        if (
            position_error > self.settings.position_tolerance_m
            or rotation_error > self.settings.rotation_tolerance_rad
        ):
            return None
        cost = joint_motion_cost(q_rad, q_start)
        return IKCandidate(q_rad.copy(), cost, position_error, rotation_error)

    def _retain_candidate(
        self, candidates: list[IKCandidate], candidate: IKCandidate,
    ) -> list[IKCandidate]:
        r"""비용이 작은 후보부터 중복을 걸러 정해진 개수만 남긴다.

        $$
        \|q_i-q_j\|_\infty\le\delta\ \Rightarrow\ \text{중복 후보}
        $$

        delta는 duplicate_tolerance_rad로, 모든 관절의 차이가 이 값 이하일 때 묶는다.
        관절 제한이 있으므로 한 바퀴 차이를 같은 각도로 감아 비교하지 않는다.
        """
        retained = []
        for item in sorted([*candidates, candidate], key=lambda item: item.cost):
            # 보관된 후보 중 모든 관절각이 가까운 후보의 존재 여부: $$d_i=\exists j:\|q_i-q_j\|_\infty\le\delta$$
            duplicate = any(
                np.allclose(item.q_rad, other.q_rad, rtol=0, atol=self.settings.duplicate_tolerance_rad)
                for other in retained
            )
            if not duplicate:
                retained.append(item)
            if len(retained) == self.settings.max_candidates:
                break
        return retained

    def solve(self, T_tip_L_tip_R_goal: ArrayLike, q_start_rad: ArrayLike) -> IKResult:
        r"""주어진 팁 상대 목표를 만족하는 해 중 관절각 변화 비용이 작은 후보를 찾는다.

        $$
        \min_q\ \tfrac12\|q-q_s\|^2
        \quad\text{subject to}\quad c(q)=0,\quad q_{\min}\le q\le q_{\max}
        $$

        q_s는 q_start_rad이며 계산 시작값이 달라져도 이 비용 기준은 유지한다.
        T_tip_L_tip_R_goal은 L팁 좌표계에서 표현한 R팁 목표 동차변환이다.
        첫 시도는 실제 시작 관절각에서, 나머지는 모든 관절 범위의 균등 표본에서 시작한다.
        제약 자코비안은 SLSQP의 수치 미분을 사용한다.
        파일이나 중간 관절각 이력은 저장하지 않으며 서로 다른 상위 후보만 보관한다.
        """
        target = as_pose(T_tip_L_tip_R_goal)
        q_start = as_joint_angles(q_start_rad)
        lower = self._limits[:, 0]
        upper = self._limits[:, 1]
        if np.any(q_start < lower) or np.any(q_start > upper):
            raise ValueError("실제 시작 관절각이 XML의 관절 범위를 벗어났습니다.")

        initial = self._validated_candidate(q_start, target, q_start)
        if initial is not None:
            return IKResult(
                True, initial.q_rad.copy(), initial.cost,
                initial.position_error_m, initial.rotation_error_rad, 0, 0,
                "현재 관절각이 이미 목표 허용오차를 만족합니다. 변화 비용은 0입니다.",
                (initial,),
            )

        random = np.random.default_rng(self.settings.random_seed)
        candidates: list[IKCandidate] = []
        solved_attempts = 0
        attempts = 0
        for attempt in range(self.settings.starts):
            attempts += 1
            seed = q_start.copy() if attempt == 0 else random.uniform(lower, upper)
            optimized = minimize(
                joint_motion_cost,
                seed,
                args=(q_start,),
                jac=joint_motion_gradient,
                method="SLSQP",
                bounds=self._limits,
                constraints={"type": "eq", "fun": self._pose_constraint, "args": (target,)},
                options={
                    "maxiter": self.settings.max_iterations,
                    "ftol": self.settings.optimizer_tolerance,
                },
            )
            if not optimized.success:
                continue
            candidate = self._validated_candidate(optimized.x, target, q_start)
            if candidate is None:
                continue
            solved_attempts += 1
            candidates = self._retain_candidate(candidates, candidate)

        if not candidates:
            return IKResult(
                False, None, None, None, None, attempts, 0,
                "설정한 시도 안에서 검증된 해를 찾지 못했습니다. 목표가 불가능하다는 증명은 아닙니다.",
            )
        best = candidates[0]
        return IKResult(
            True, best.q_rad.copy(), best.cost,
            best.position_error_m, best.rotation_error_rad, attempts, solved_attempts,
            "검증된 서로 다른 후보를 비용이 작은 순서로 정렬했습니다.",
            tuple(candidates),
        )
