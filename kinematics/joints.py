"""팔 관절의 공통 순서와 관절각 입력 형식을 정의한다."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


ARM_JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6", "J7")


def as_joint_angles(q_rad: ArrayLike) -> NDArray[np.float64]:
    """일곱 관절의 라디안 입력을 검사하고 독립적인 배열로 반환한다."""
    if np.iscomplexobj(q_rad):
        raise ValueError("관절각은 복소수가 아닌 실수여야 합니다.")
    angles = np.array(q_rad, dtype=float, copy=True)
    if angles.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError("관절각은 J1부터 J7 순서의 (7,) 배열이어야 합니다.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("관절각에 NaN이나 무한대를 넣을 수 없습니다.")
    return angles
