"""XML 모델과 팔 관절각으로 양쪽 팁의 위치와 방향을 계산한다.
MuJoCo의 순기구학을 사용하며 뷰어와 물리 시간 진행에는 관여하지 않는다.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from kinematics.joints import ARM_JOINT_NAMES, as_joint_angles


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xs" / "model.xml"


@dataclass
class FKResult:
    r"""FK 계산의 반환 형식으로, 아래 세 가지 자세를 담는다.

    T_world_tip_L: MuJoCo 월드 좌표계에서 본 왼쪽 팁의 위치와 방향.
    T_world_tip_R: MuJoCo 월드 좌표계에서 본 오른쪽 팁의 위치와 방향.
    T_tip_L_tip_R: 왼쪽 팁 좌표계에서 본 오른쪽 팁의 위치와 방향.

    각 자세는 \(4 \times 4\) 동차변환 행렬로 표현한다.
    [:3, 3]은 기준 좌표계로 표현한 대상 팁의 원점 위치(m)이고,
    [:3, :3]은 기준 좌표계로 표현한 대상 팁의 방향(회전행렬)이다.
    회전행렬의 각 열은 대상 팁의 X, Y, Z축 방향에 해당한다.
    반환 배열은 다음 FK 계산에 의해 바뀌지 않는다.
    """

    T_world_tip_L: NDArray[np.float64]
    T_world_tip_R: NDArray[np.float64]
    T_tip_L_tip_R: NDArray[np.float64]


def _site_transform(data: mujoco.MjData, site_id: int) -> NDArray[np.float64]:
    r"""MuJoCo 사이트의 월드 위치와 방향을 동차변환으로 묶는다.

    $$
    {}^W T_S = \begin{bmatrix} {}^W R_S & {}^W p_S \\ 0 & 1 \end{bmatrix}
    $$

    W는 월드, S는 사이트 좌표계이며 R은 방향, p는 원점 위치이다.
    """
    # 동차변환의 초기 행렬: $$T = I_4$$
    transform = np.eye(4)
    transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[site_id]
    return transform


def _relative_transform(
    T_world_tip_L: NDArray[np.float64],
    T_world_tip_R: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""양쪽 팁의 월드 자세를 왼쪽 팁 기준의 상대 자세로 변환한다.

    $$
    {}^L T_R = ({}^W T_L)^{-1} {}^W T_R
    = \begin{bmatrix}
      ({}^W R_L)^T {}^W R_R & ({}^W R_L)^T ({}^W p_R - {}^W p_L) \\
      0 & 1
    \end{bmatrix}
    $$

    W는 월드, L과 R은 tip_L과 tip_R 좌표계이다.
    R은 회전행렬, p는 위치 벡터, T는 동차변환이다.
    """
    R_world_tip_L = T_world_tip_L[:3, :3]
    R_world_tip_R = T_world_tip_R[:3, :3]
    p_world_tip_L = T_world_tip_L[:3, 3]
    p_world_tip_R = T_world_tip_R[:3, 3]

    # 왼쪽 팁 기준의 오른쪽 팁 방향: $${}^L R_R = ({}^W R_L)^T {}^W R_R$$
    R_tip_L_tip_R = R_world_tip_L.T @ R_world_tip_R
    # 월드 기준의 두 팁 사이 변위: $$\Delta p_W = {}^W p_R - {}^W p_L$$
    displacement_world = p_world_tip_R - p_world_tip_L
    # 변위를 왼쪽 팁 좌표계로 표현: $${}^L p_R = ({}^W R_L)^T \Delta p_W$$
    p_tip_L_tip_R = R_world_tip_L.T @ displacement_world

    # 동차변환의 초기 행렬: $${}^L T_R = I_4$$
    T_tip_L_tip_R = np.eye(4)
    T_tip_L_tip_R[:3, :3] = R_tip_L_tip_R
    T_tip_L_tip_R[:3, 3] = p_tip_L_tip_R
    return T_tip_L_tip_R


class ForwardKinematics:
    """모델을 한 번 읽고 여러 관절각에 대한 FK를 반복 계산한다.

    XML의 위치·방향·회전축을 그대로 사용한다.
    실물 모터 보정값을 적용하거나 관절각을 제한 범위로 잘라내지 않는다.
    월드 출력은 XML 배치 기준이며, 다른 고정점 배치는 TipAnchor.place에서 적용한다.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """모델을 읽고 팔 관절의 상태 주소와 양쪽 팁의 식별자를 준비한다."""
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

        joint_ids = [self._model.joint(name).id for name in ARM_JOINT_NAMES]
        if np.any(self._model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("J1부터 J7까지는 모두 회전 관절이어야 합니다.")
        self._qpos_indices = self._model.jnt_qposadr[joint_ids].copy()
        self._joint_ids = np.asarray(joint_ids)
        self._tip_L_id = self._model.site("tip_L").id
        self._tip_R_id = self._model.site("tip_R").id

    @property
    def joint_limits(self) -> NDArray[np.float64]:
        """XML에 지정된 팔 관절의 하한·상한을 관절 순서대로 복사해 반환한다.

        각 행은 해당 관절의 하한과 상한(rad)이다.
        XML에서 제한이 꺼진 관절은 음의 무한대와 양의 무한대로 표현한다.
        """
        limits = self._model.jnt_range[self._joint_ids].copy()
        unlimited = ~self._model.jnt_limited[self._joint_ids].astype(bool)
        limits[unlimited] = [-np.inf, np.inf]
        return limits

    def forward(self, q_rad: ArrayLike) -> FKResult:
        """팔 관절각을 적용하고 양쪽 팁의 월드 자세와 상대 자세를 반환한다.

        q_rad는 J1, J2, J3, J4, J5, J6, J7 순서의 라디안 배열이며
        형상은 (7,)이어야 한다. 모든 값은 유한한 실수여야 한다.
        G_L과 G_R은 XML의 초기값을 유지하며 팁 사이 FK에 포함하지 않는다.
        """
        q_rad = as_joint_angles(q_rad)

        self._data.qpos[:] = self._model.qpos0
        self._data.qpos[self._qpos_indices] = q_rad
        mujoco.mj_kinematics(self._model, self._data)

        T_world_tip_L = _site_transform(self._data, self._tip_L_id)
        T_world_tip_R = _site_transform(self._data, self._tip_R_id)
        T_tip_L_tip_R = _relative_transform(T_world_tip_L, T_world_tip_R)
        return FKResult(T_world_tip_L, T_world_tip_R, T_tip_L_tip_R)
