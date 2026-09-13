"""고정한 팁의 월드 자세를 기준으로 로봇 배치와 IK 상대 목표를 계산한다.
XML 기준 FK와 실제 고정 위치를 분리하며 관절각·모델·최적화 알고리즘은 변경하지 않는다.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kinematics.fk import FKResult
from kinematics.poses import as_pose, invert_pose


@dataclass(frozen=True)
class TipAnchor:
    """어느 팁을 월드의 어느 위치와 방향에 고정할지 지정한다.

    fixed_tip은 tip_L 또는 tip_R이다.
    T_world_fixed_tip은 잡은 순간의 팁 월드 자세로, 이후 관절각이 바뀌어도 유지된다.
    입력 행렬은 복사해서 읽기 전용으로 보관한다. 고정점을 바꿀 때는 새 객체를 만든다.
    """

    fixed_tip: Literal["tip_L", "tip_R"]
    T_world_fixed_tip: NDArray[np.float64]

    def __post_init__(self) -> None:
        """팁 이름과 고정 자세를 검사하고 외부 입력의 수정으로부터 분리한다."""
        if self.fixed_tip not in ("tip_L", "tip_R"):
            raise ValueError("고정 팁은 tip_L 또는 tip_R이어야 합니다.")
        pose = as_pose(self.T_world_fixed_tip)
        pose.setflags(write=False)
        object.__setattr__(self, "T_world_fixed_tip", pose)

    def world_from_model(self, model_fk: FKResult) -> NDArray[np.float64]:
        r"""XML 배치의 로봇을 고정 팁에 맞춰 옮기는 월드 변환을 반환한다.

        $$
        {}^W T_M={}^W T_F\left({}^M T_F(q)\right)^{-1}
        $$

        W는 실제 월드, M은 XML의 기준 배치, F는 고정 팁이다.
        model_fk에는 고정점 배치를 적용하기 전의 ForwardKinematics.forward 결과를 넣는다.
        반환 행렬은 로봇의 모든 링크·팁에 공통 적용할 배치이며 빔에는 적용하지 않는다.
        """
        if self.fixed_tip == "tip_L":
            T_model_fixed_tip = model_fk.T_world_tip_L
        else:
            T_model_fixed_tip = model_fk.T_world_tip_R
        T_fixed_tip_model = invert_pose(T_model_fixed_tip)
        # 고정 팁을 거쳐 XML 배치를 실제 월드로 연결: $${}^W T_M={}^W T_F\,{}^F T_M$$
        T_world_model = self.T_world_fixed_tip @ T_fixed_tip_model
        return T_world_model

    def place(self, model_fk: FKResult) -> FKResult:
        r"""XML 기준 FK 결과를 고정 팁이 유지되는 양쪽 팁의 월드 자세로 바꾼다.

        $$
        {}^W T_S={}^W T_M\,{}^M T_S,\qquad S\in\{L,R\}
        $$

        W는 실제 월드, M은 XML 배치, S는 배치를 구할 팁이다.
        model_fk는 ForwardKinematics.forward가 직접 반환한 값이다.
        반환 형식은 FKResult이며 두 팁의 상대 자세와 관절각 정의는 유지된다.
        """
        T_world_model = self.world_from_model(model_fk)
        # 실제 월드에서 본 왼쪽 팁: $${}^W T_L={}^W T_M\,{}^M T_L$$
        T_world_tip_L = T_world_model @ model_fk.T_world_tip_L
        # 실제 월드에서 본 오른쪽 팁: $${}^W T_R={}^W T_M\,{}^M T_R$$
        T_world_tip_R = T_world_model @ model_fk.T_world_tip_R
        return FKResult(T_world_tip_L, T_world_tip_R, model_fk.T_tip_L_tip_R.copy())

    def to_relative_target(self, T_world_moving_tip_goal: ArrayLike) -> NDArray[np.float64]:
        r"""움직일 팁의 월드 목표를 기존 IK가 받는 L팁 기준 R팁 목표로 변환한다.

        $$
        {}^L T_{R,d}=\left({}^W T_{L,d}\right)^{-1}{}^W T_{R,d}
        $$

        W는 실제 월드이고 첨자 d는 목표 자세이다.
        입력은 L 고정일 때 R팁 목표, R 고정일 때 L팁 목표이다.
        고정 팁의 목표에는 저장된 고정 자세를 사용한다.
        반환값은 항상 InverseKinematics.solve의 첫 인자에 그대로 전달할 수 있다.
        """
        moving_goal = as_pose(T_world_moving_tip_goal)
        if self.fixed_tip == "tip_L":
            T_world_tip_L_goal = self.T_world_fixed_tip
            T_world_tip_R_goal = moving_goal
        else:
            T_world_tip_L_goal = moving_goal
            T_world_tip_R_goal = self.T_world_fixed_tip
        T_tip_L_world_goal = invert_pose(T_world_tip_L_goal)
        # 월드 목표를 L팁 기준 R팁 목표로 변환: $${}^L T_{R,d}={}^L T_{W,d}\,{}^W T_{R,d}$$
        T_tip_L_tip_R_goal = T_tip_L_world_goal @ T_world_tip_R_goal
        return T_tip_L_tip_R_goal
