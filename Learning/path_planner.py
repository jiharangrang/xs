"""기존 직선 IK에 그리퍼·빔 거리 제약을 넣어 같은 6단계 경로를 새로 생성한다.
메시와 SE3NN은 같은 최적화·유한차분·보간 검사 조건에서 거리 계산만 교체한다.
"""

from dataclasses import dataclass
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

from common import REPO

sys.path.insert(0, str(REPO))
from kinematics.ik import IKResult, IKSettings, InverseKinematics, joint_motion_cost, joint_motion_gradient
from kinematics.joints import as_joint_angles
from kinematics.poses import as_pose


@dataclass(frozen=True)
class PathSettings:
    """두 거리 계산 방식에 공통으로 적용할 간격과 표본·미분 간격을 보관한다."""

    margin_m: float = .0005
    difference_step_rad: float = .0001
    edge_subdivisions: int = 4
    validation_subdivisions: int = 8
    max_iterations: int = 100
    jaw_rad: float = np.deg2rad(-120.)

    def __post_init__(self):
        """비교에 필요한 간격과 반복 예산의 유효성을 확인한다."""
        for value in (self.margin_m, self.difference_step_rad):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("거리 여유와 수치 미분 간격은 유한한 양수여야 합니다.")
        for value in (self.edge_subdivisions, self.validation_subdivisions, self.max_iterations):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("보간 분할과 반복 횟수는 양의 정수여야 합니다.")
        if self.validation_subdivisions < self.edge_subdivisions or not np.isfinite(self.jaw_rad):
            raise ValueError("최종 검사 밀도와 턱 각도를 확인하세요.")


def dense_path(q, subdivisions):
    r"""관절 경로의 각 구간을 같은 개수로 나누고 마지막 지점을 한 번 포함한다.

    $$q_i(u)=q_i+u(q_{i+1}-q_i),\qquad u\in\{0,1/s,\ldots,(s-1)/s\}$$
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or len(q) < 1 or subdivisions < 1:
        raise ValueError("비어 있지 않은 관절 경로와 양의 분할 수가 필요합니다.")
    if len(q) == 1:
        return q.copy()
    fractions = np.arange(subdivisions) / subdivisions
    # 인접 관절각의 선형 보간: $$q_i(u)=q_i+u(q_{i+1}-q_i)$$
    points = q[:-1, None, :] + fractions[None, :, None] * (q[1:] - q[:-1])[:, None, :]
    return np.concatenate((points.reshape(-1, q.shape[1]), q[-1:]))


class DistanceOracle:
    """같은 자세 묶음에 메시 또는 CPU 신경망 거리를 반환하고 실제 비용을 기록한다."""

    def __init__(self, scene, model=None):
        """검증 장면과 선택적인 학습 모델을 연결한다."""
        self.scene = scene
        self.model = model
        if model is not None:
            model.cpu().eval()
        self.reset()

    def reset(self):
        """새 경로 요청의 거리 질의 통계를 초기화한다."""
        self.calls = 0
        self.configurations = 0
        self.elapsed_s = 0.

    def __call__(self, q):
        """끝점·보간점·미분점에 공통인 순서로 거리 배열을 반환한다."""
        began = time.perf_counter()
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 8 or not np.isfinite(q).all():
            raise ValueError("거리 입력은 유한한 관절 여덟 개의 묶음이어야 합니다.")
        if np.any(q < self.scene.limits[:, 0] - 1e-7) or np.any(q > self.scene.limits[:, 1] + 1e-7):
            raise ValueError("거리 질의가 관절 제한을 벗어났습니다.")
        if self.model is None:
            values = self.scene.label(q).astype(np.float64)
        else:
            with torch.inference_mode():
                values = self.model(torch.as_tensor(q, dtype=torch.float32)).numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("거리 예측이 유한하지 않습니다.")
        self.calls += 1
        self.configurations += len(q)
        self.elapsed_s += time.perf_counter() - began
        return values


class CollisionAwareIK(InverseKinematics):
    """기존 IK의 자세·관절 제약에 보간 구간의 거리 부등식을 추가한다."""

    def __init__(self, oracle, settings=None):
        """동일한 기구학 솔버와 경로 거리 조건을 준비한다."""
        self.path_settings = PathSettings() if settings is None else settings
        super().__init__(settings=IKSettings(starts=1, max_iterations=self.path_settings.max_iterations))
        self.oracle = oracle
        self.solve_calls = 0
        self.optimizer_iterations = 0

    def with_jaw(self, q):
        """팔 관절 묶음에 일정한 오른쪽 턱 각도를 붙인다."""
        q = np.asarray(q)
        return np.column_stack((q, np.full(len(q), self.path_settings.jaw_rad)))

    def guard_segment(self, q):
        """관절 보간으로 만드는 첫 단계도 같은 거리 조건을 통과해야 허용한다."""
        dense = dense_path(q, self.path_settings.edge_subdivisions)
        distance = self.oracle(self.with_jaw(dense))
        if np.min(distance) < self.path_settings.margin_m - 1e-7:
            raise RuntimeError(f"첫 관절 보간 구간의 거리 조건 실패: {distance.min():.6g} m")

    def edge_distances(self, endpoints, start):
        r"""각 후보 끝점까지의 관절 보간 구간에서 거리 조건을 평가한다.

        $$q_j(u)=q_s+u(q_j-q_s),\qquad u\in\{1/s,\ldots,1\}$$
        """
        fractions = np.arange(1, self.path_settings.edge_subdivisions + 1) / self.path_settings.edge_subdivisions
        # 같은 출발점에서 각 후보까지의 구간 표본: $$q_j(u)=q_s+u(q_j-q_s)$$
        points = start + fractions[None, :, None] * (endpoints - start)[:, None, :]
        values = self.oracle(self.with_jaw(points.reshape(-1, 7)))
        return values.reshape(len(endpoints), self.path_settings.edge_subdivisions)

    def edge_jacobian(self, endpoint, start):
        r"""두 계산 방식 모두 같은 간격의 중앙차분으로 거리 제약 자코비안을 구한다.

        $$J_{ij}\simeq\frac{d_i(q_j^+)-d_i(q_j^-)}{(q_j^+)_j-(q_j^-)_j}\frac1\ell$$

        관절 경계에서는 가능한 쪽의 차분을 사용하고, ell은 기존 IK의 위치 척도다.
        """
        delta = np.eye(7) * self.path_settings.difference_step_rad
        plus = np.clip(endpoint + delta, self._limits[:, 0], self._limits[:, 1])
        minus = np.clip(endpoint - delta, self._limits[:, 0], self._limits[:, 1])
        values = self.edge_distances(np.concatenate((plus, minus)), start)
        # 관절 제한을 반영한 실제 차분 폭: $$h_j=(q_j^+)_j-(q_j^-)_j$$
        width = np.diag(plus - minus)
        # 거리 함수의 동일한 유한차분: $$D_{ji}=(d_i(q_j^+)-d_i(q_j^-))/h_j$$
        derivative = (values[:7] - values[7:]) / width[:, None]
        # 기존 위치 제약과 같은 척도로 변환: $$J=D^T/\ell$$
        return derivative.T / self.settings.position_scale_m

    def solve(self, T_tip_L_tip_R_goal, q_start_rad):
        r"""자세 일치와 구간 거리 조건을 동시에 만족하는 가까운 관절 해를 찾는다.

        $$\min_q\tfrac12\|q-q_s\|^2\quad\text{s.t.}\quad c(q)=0,\ d(q_s+u(q-q_s))\ge m$$
        """
        self.solve_calls += 1
        target = as_pose(T_tip_L_tip_R_goal)
        start = as_joint_angles(q_start_rad)
        if np.any(start < self._limits[:, 0]) or np.any(start > self._limits[:, 1]):
            raise ValueError("IK 시작 관절각이 제한을 벗어났습니다.")
        cached_q, cached_d = None, None

        def distances(q):
            """동일한 최적화 지점의 반복 질의만 공통으로 재사용한다."""
            nonlocal cached_q, cached_d
            if cached_q is None or not np.array_equal(q, cached_q):
                cached_d = self.edge_distances(q[None], start)[0]
                cached_q = q.copy()
            return cached_d

        def clearance(q):
            r"""최소 여유 이상의 거리를 비음수 부등식으로 표현한다.

            $$g(q)=(d(q)-m)/\ell$$
            """
            # 미터 거리 조건의 척도 변환: $$g=(d-m)/\ell$$
            return (distances(q) - self.path_settings.margin_m) / self.settings.position_scale_m

        if np.min(distances(start)) < self.path_settings.margin_m - 1e-7:
            return IKResult(False, None, None, None, None, 0, 0, "출발 자세의 거리 조건 실패")
        optimized = minimize(joint_motion_cost, start, args=(start,), jac=joint_motion_gradient,
            method="SLSQP", bounds=self._limits,
            constraints=[{"type": "eq", "fun": self._pose_constraint, "args": (target,)},
                         {"type": "ineq", "fun": clearance, "jac": lambda q: self.edge_jacobian(q, start)}],
            options={"maxiter": self.settings.max_iterations, "ftol": self.settings.optimizer_tolerance})
        self.optimizer_iterations += int(optimized.nit)
        candidate = self._validated_candidate(optimized.x, target, start) if optimized.success else None
        if candidate is None or np.min(distances(optimized.x)) < self.path_settings.margin_m - 1e-7:
            return IKResult(False, None, None, None, None, 1, 0, f"거리 제약 IK 실패: {optimized.message}")
        return IKResult(True, candidate.q_rad.copy(), candidate.cost, candidate.position_error_m,
                        candidate.rotation_error_rad, 1, 1, "자세와 구간 거리 조건 통과", (candidate,))
