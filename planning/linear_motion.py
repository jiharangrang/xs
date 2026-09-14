"""한쪽 팁을 고정하고 반대 팁의 방향을 유지하며 직선 이동 관절 경로를 만든다.
기존 IK를 순차 호출하고 관절각 보간 구간을 FK로 표본 검사한다.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kinematics.anchoring import TipAnchor
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import as_joint_angles
from kinematics.poses import pose_error


@dataclass
class LinearPathResult:
    r"""직선 이동 계산의 성공 여부, 관절 경로와 목표 자세를 담는다.

    q_path_rad: 시작점을 포함한 \((K+1, 7)\) 관절각 배열이며 순서는 J1부터 J7, 단위는 rad이다.
        실패하면 None이며 중간까지 계산한 경로를 완성 경로로 반환하지 않는다.
    target_poses_world: 각 지점에서 월드 기준으로 요구한 이동 팁 자세의 \((K+1,4,4)\) 배열이다.
    max_position_error_m, max_rotation_error_rad: 검사한 표본에서 목표 대비 가장 큰 오차이다.
    max_joint_step_rad: 검사한 인접 지점 사이에서 한 관절이 가장 크게 변한 양이다.
    failed_step: 실패한 구간 번호이며 첫 구간은 1, 성공하면 None이다.
    message: 성공 또는 중단 이유이다. 실패 시 오차 통계는 중단 시점까지의 값이다.

    K는 이동 구간 수이다. 시간·속도 계획, 충돌 검증과 전체 보행 최적화는 포함하지 않는다.
    """

    success: bool
    q_path_rad: NDArray[np.float64] | None
    target_poses_world: NDArray[np.float64]
    max_position_error_m: float = 0.0
    max_rotation_error_rad: float = 0.0
    max_joint_step_rad: float = 0.0
    failed_step: int | None = None
    message: str = ""


def plan_linear_motion(
    q_start_rad: ArrayLike,
    anchor: TipAnchor,
    displacement_world_m: ArrayLike,
    *,
    steps: int = 20,
    max_joint_step_rad: float = 0.2,
    solver: InverseKinematics | None = None,
) -> LinearPathResult:
    r"""고정 팁 반대편의 팁을 월드 변위만큼 직선으로 옮기는 관절 경로를 계산한다.

    $$
    p_k=p_0+\frac{k}{K}\Delta p_W,\qquad R_k=R_0
    $$

    \(p_0,R_0\)는 고정점 배치를 적용한 시작 팁의 위치와 방향이며,
    \(\Delta p_W\)는 displacement_world_m, \(K\)는 steps이다.
    q_start_rad는 현재 일곱 관절각, anchor는 움직이지 않을 팁의 월드 자세이다.
    max_joint_step_rad는 구간당 각도 급변을 거부하는 계획 기준이며 모터 사양이 아니다.
    solver를 생략하면 직전 관절각 하나에서 시작하는 기존 IK를 사용한다.

    각 구간의 사분점과 끝점에서 FK 오차를 확인한다. 유한한 표본 검사이며,
    연속 구간 전체의 오차 상한을 증명하지 않는다. 허용오차는 solver.settings를 따른다.
    입력 오류는 ValueError, IK·연속성·오차 검사 실패는 success=False로 반환한다.
    """
    q_start = as_joint_angles(q_start_rad)
    if np.iscomplexobj(displacement_world_m):
        raise ValueError("월드 변위는 복소수가 아닌 실수여야 합니다.")
    displacement = np.array(displacement_world_m, dtype=float, copy=True)
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise ValueError("월드 변위는 미터 단위의 유한한 (3,) 배열이어야 합니다.")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps는 1 이상의 정수여야 합니다.")
    if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0:
        raise ValueError("max_joint_step_rad는 유한한 양수여야 합니다.")

    if solver is None:
        solver = InverseKinematics(settings=IKSettings(starts=1))
    limits = solver.fk.joint_limits
    if np.any(q_start < limits[:, 0]) or np.any(q_start > limits[:, 1]):
        raise ValueError("시작 관절각이 캘리브레이션의 관절 범위를 벗어났습니다.")
    placed_start = anchor.place(solver.fk.forward(q_start))
    moving_attribute = "T_world_tip_L" if anchor.fixed_tip == "tip_R" else "T_world_tip_R"
    start_pose = getattr(placed_start, moving_attribute)
    targets = np.repeat(start_pose[None, :, :], steps + 1, axis=0)
    for index in range(1, steps + 1):
        # 전체 직선에서 해당 지점의 진행 비율: $$s_k=k/K$$
        fraction = index / steps
        # 시작 방향을 유지한 채 월드 위치만 이동: $$p_k=p_0+s_k\Delta p_W$$
        targets[index, :3, 3] = start_pose[:3, 3] + fraction * displacement

    result = LinearPathResult(False, None, targets)
    path = [q_start]
    for index in range(1, steps + 1):
        relative_target = anchor.to_relative_target(targets[index])
        ik_result = solver.solve(relative_target, path[-1])
        if not ik_result.success:
            result.failed_step = index
            result.message = f"{index}/{steps} 구간 IK 실패: {ik_result.message}"
            return result
        q_next = ik_result.q_rad
        # 인접 지점 사이의 최대 관절각 변화: $$d_q=\max_j|q_{k,j}-q_{k-1,j}|$$
        joint_step = float(np.max(np.abs(q_next - path[-1])))
        result.max_joint_step_rad = max(result.max_joint_step_rad, joint_step)
        if joint_step > max_joint_step_rad:
            result.failed_step = index
            result.message = (
                f"{index}/{steps} 구간 관절각 급변: {joint_step:.6g} rad가 "
                f"허용값 {max_joint_step_rad:.6g} rad를 넘었습니다."
            )
            return result
        position_error, rotation_error = _segment_errors(
            solver, anchor, moving_attribute, path[-1], q_next,
            targets[index - 1], targets[index],
        )
        result.max_position_error_m = max(result.max_position_error_m, position_error)
        result.max_rotation_error_rad = max(result.max_rotation_error_rad, rotation_error)
        if (
            position_error > solver.settings.position_tolerance_m
            or rotation_error > solver.settings.rotation_tolerance_rad
        ):
            result.failed_step = index
            result.message = (
                f"{index}/{steps} 구간 FK 오차 초과: 위치 {position_error:.6g} m, "
                f"방향 {rotation_error:.6g} rad. 구간을 더 잘게 나누거나 시작 자세를 바꿔야 합니다."
            )
            return result
        path.append(q_next.copy())

    result.success = True
    result.q_path_rad = np.array(path)
    result.message = f"직선 이동 {steps}구간의 IK와 구간별 FK 표본 검사를 통과했습니다."
    return result


def _segment_errors(
    solver: InverseKinematics,
    anchor: TipAnchor,
    moving_attribute: str,
    q_before: NDArray[np.float64],
    q_after: NDArray[np.float64],
    target_before: NDArray[np.float64],
    target_after: NDArray[np.float64],
) -> tuple[float, float]:
    r"""관절각 보간으로 생기는 실제 팁 자세와 직선 목표 사이의 최대 표본 오차를 구한다.

    $$
    q(u)=(1-u)q_a+u q_b,\qquad p_d(u)=(1-u)p_a+u p_b
    $$

    \(u\)는 한 구간 내의 비율이며 사분점과 끝점에서 검사한다.
    반환값은 위치 오차의 최대 노름(m)과 방향 오차의 최대 회전각(rad)이다.
    """
    max_position_error = 0.0
    max_rotation_error = 0.0
    for fraction in (0.25, 0.5, 0.75, 1.0):
        # 두 지점 사이의 관절각 선형 보간: $$q(u)=(1-u)q_a+u q_b$$
        q_sample = (1 - fraction) * q_before + fraction * q_after
        placed = anchor.place(solver.fk.forward(q_sample))
        actual = getattr(placed, moving_attribute)
        target = target_before.copy()
        # 같은 비율에서의 직선 목표 위치: $$p_d(u)=(1-u)p_a+u p_b$$
        target[:3, 3] = (1 - fraction) * target_before[:3, 3] + fraction * target_after[:3, 3]
        error = pose_error(actual, target)
        # 위치 오차 벡터의 길이: $$e_p=\|p-p_d\|_2$$
        position_error = float(np.linalg.norm(error[:3]))
        # 회전벡터의 길이가 나타내는 방향 오차: $$e_R=\|\operatorname{Log}(R_d^TR)^\vee\|_2$$
        rotation_error = float(np.linalg.norm(error[3:]))
        max_position_error = max(max_position_error, position_error)
        max_rotation_error = max(max_rotation_error, rotation_error)
    return max_position_error, max_rotation_error
