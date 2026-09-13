"""그리퍼와 빔의 작업 조건을 IK에 전달할 목표 자세로 만든다."""

import numpy as np
from numpy.typing import NDArray


def beam_grasp_target(distance_m: float = 0.10) -> NDArray[np.float64]:
    r"""현재 빔 배치에서 L팁보다 앞쪽을 잡는 R팁의 상대 목표 자세를 만든다.

    $$
    {}^L T_{R,d} =
    \begin{bmatrix}-1&0&0&-s\\0&-1&0&0\\0&0&1&0\\0&0&0&1\end{bmatrix}
    $$

    s는 distance_m으로 전달한 두 팁의 빔 길이 방향 간격(m)이다.
    현재 모델의 전진 방향인 월드 +X는 L팁의 -X 방향이다.
    두 팁은 같은 높이와 횡방향 위치에 놓이고, X·Y축은 반대, Z축은 같다.
    반환값은 월드 자세가 아닌 L팁 기준 R팁 목표 자세이다.
    """
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("빔 전진 거리는 유한한 양수여야 합니다.")
    # 목표 방향과 동차좌표의 마지막 성분: $${}^L T_{R,d}(0)=\operatorname{diag}(-1,-1,1,1)$$
    target = np.diag([-1.0, -1.0, 1.0, 1.0])
    # L팁의 음의 X축을 따라 목표 위치 지정: $${}^L p_{R,d,x}=-s$$
    target[0, 3] = -distance_m
    return target
