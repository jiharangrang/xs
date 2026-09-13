"""동차변환 입력을 검사하고 같은 기준 좌표계에서 두 자세의 오차를 계산한다."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def as_pose(transform: ArrayLike) -> NDArray[np.float64]:
    """유효한 강체 동차변환인지 검사하고 독립적인 배열로 반환한다."""
    if np.iscomplexobj(transform):
        raise ValueError("자세는 복소수가 아닌 실수여야 합니다.")
    pose = np.array(transform, dtype=float, copy=True)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("자세는 유한한 실수로 된 (4, 4) 행렬이어야 합니다.")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("동차변환의 마지막 행은 [0, 0, 0, 1]이어야 합니다.")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("회전행렬의 축들은 서로 수직인 단위 벡터여야 합니다.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0):
        raise ValueError("회전행렬은 오른손 좌표계여야 합니다.")
    return pose


def pose_error(actual: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""현재 자세와 목표 자세의 위치 오차와 회전벡터 오차를 반환한다.

    $$
    e = \begin{bmatrix}p-p_d\\ \operatorname{Log}(R_d^T R)^\vee\end{bmatrix}
    $$

    p와 R은 actual의 위치·방향, 첨자 d는 target의 목표값이다.
    두 입력은 같은 기준 좌표계에서 표현한 유효한 동차변환이어야 한다.
    반환값의 앞 세 성분은 위치 오차(m), 뒤 세 성분은 회전 오차(rad)이다.
    회전벡터는 목표 프레임 기준이며 그 길이가 두 방향 사이의 각도이다.
    """
    # 기준 좌표계로 표현한 위치 차이: $$e_p = p-p_d$$
    position_error = actual[:3, 3] - target[:3, 3]
    # 목표 방향에서 본 현재 방향: $$R_e = R_d^T R$$
    rotation_error = target[:3, :3].T @ actual[:3, :3]
    # 방향 차이를 축과 회전각으로 표현: $$e_R = \operatorname{Log}(R_e)^\vee$$
    rotation_vector = Rotation.from_matrix(rotation_error).as_rotvec()
    return np.concatenate((position_error, rotation_vector))
