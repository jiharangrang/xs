"""표시용 MuJoCo 모델을 준비하고 입력 관절각을 그대로 반영한다.
모델 XML과 모터 설정을 수정하거나 물리 시간을 진행하지 않는다.
"""

from pathlib import Path

import mujoco
from numpy.typing import ArrayLike

from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import ARM_JOINT_NAMES, as_joint_angles


def set_arm_angles(model: mujoco.MjModel, data: mujoco.MjData, q_rad: ArrayLike) -> None:
    """J1부터 J7까지의 입력 각도를 자르지 않고 적용한 뒤 기구학 상태를 갱신한다."""
    angles = as_joint_angles(q_rad)
    for name, angle in zip(ARM_JOINT_NAMES, angles, strict=True):
        data.joint(name).qpos[0] = angle
    mujoco.mj_forward(model, data)


def load_model(
    q_rad: ArrayLike = (0.0,) * 7, model_path: str | Path = DEFAULT_MODEL_PATH
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """관절 좌표축 표시를 붙인 모델과 지정된 관절각의 상태를 반환한다."""
    angles = as_joint_angles(q_rad)
    spec = mujoco.MjSpec.from_file(str(model_path))
    for joint in spec.joints:
        # 표시용 사이트는 메모리에만 추가하며 원본 XML에는 저장하지 않는다.
        joint.parent.add_site(
            name=joint.name,
            pos=joint.pos,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0002, 0, 0],
            rgba=[0.7, 0.7, 0.7, 1],
        )
    model = spec.compile()
    model.vis.scale.framelength = 0.4
    model.vis.scale.framewidth = 0.004
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_arm_angles(model, data, angles)
    return model, data
