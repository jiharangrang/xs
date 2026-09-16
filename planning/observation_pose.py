"""최적화된 파지 자세에서 오른쪽 팁의 높이만 낮춰 관측 시작 자세를 계산한다.
기존 FK·IK와 왼쪽 고정점을 사용하며 파일 저장이나 모터 구동은 하지 않는다.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kinematics.anchoring import TipAnchor
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles


@dataclass(frozen=True)
class ObservationPose:
    """파지 기준각과 관측 시작각, 오른쪽 팁의 월드 목표 및 IK 오차를 담는다."""

    q_grasp_rad: NDArray[np.float64]
    q_rad: NDArray[np.float64]
    target_world: NDArray[np.float64]
    drop_m: float
    position_error_m: float
    rotation_error_rad: float


def plan_observation_pose(
    q_grasp_rad: ArrayLike, *, drop_m: float,
    solver: InverseKinematics | None = None,
) -> ObservationPose:
    r"""왼쪽 팁을 고정하고 오른쪽 팁의 위치 중 월드 높이만 낮춘 자세를 구한다.

    $$p_{R,d}=p_{R,g}-h e_{z,W},\qquad R_{R,d}=R_{R,g}$$

    g는 입력 파지 자세, d는 관측 목표, h는 drop_m이다.
    현재 수평 빔 모델의 월드 음의 Z축을 아래로 사용한다.
    파지 자세를 초기값으로 기존 IK를 한 번 풀며 실패하면 ValueError를 발생시킨다.
    생성하는 것은 정적 자세이며 파지 자세에서 내려오는 이동 경로가 아니다.
    """
    if not np.isscalar(drop_m) or not np.isfinite(drop_m) or drop_m <= 0:
        raise ValueError("내릴 높이는 유한한 양수여야 합니다.")
    q_grasp = as_joint_angles(q_grasp_rad)
    solver = solver if solver is not None else InverseKinematics(settings=IKSettings(starts=1))
    grasp = solver.fk.forward(q_grasp)
    anchor = TipAnchor("tip_L", grasp.T_world_tip_L)
    target_world = grasp.T_world_tip_R.copy()
    # 오른쪽 팁의 월드 높이만 낮춤: $$p_{R,d,z}=p_{R,g,z}-h$$
    target_world[2, 3] -= drop_m
    result = solver.solve(anchor.to_relative_target(target_world), q_grasp)
    if not result.success:
        raise ValueError(f"관측 시작 자세를 구하지 못했습니다: {result.message}")
    return ObservationPose(
        q_grasp.copy(), result.q_rad.copy(), target_world, float(drop_m),
        result.position_error_m, result.rotation_error_rad,
    )
