"""동차변환 입력 검사, 기준 좌표계를 뒤집는 역변환과 자세 오차 계산을 제공한다."""

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


def invert_pose(transform: ArrayLike) -> NDArray[np.float64]:
    r"""강체 동차변환의 기준과 대상을 뒤집은 역행렬을 반환한다.

    $$
    T^{-1}=\begin{bmatrix}R^T&-R^Tp\\0&1\end{bmatrix}
    $$

    R과 p는 입력 transform의 회전행렬과 위치 벡터이다.
    반환값은 입력과 메모리를 공유하지 않는 동차변환이다.
    """
    pose = as_pose(transform)
    inverse = np.eye(4)
    # 반대 기준에서 표현한 회전행렬: $$R_{\mathrm{inv}}=R^T$$
    inverse[:3, :3] = pose[:3, :3].T
    # 위치 벡터를 대상 좌표계의 축으로 표현: $$p'=R^Tp$$
    rotated_position = inverse[:3, :3] @ pose[:3, 3]
    # 뒤집힌 원점 사이의 위치 벡터: $$p_{\mathrm{inv}}=-p'$$
    inverse[:3, 3] = -rotated_position
    return inverse


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
