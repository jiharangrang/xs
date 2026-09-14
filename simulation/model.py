"""표시용 MuJoCo 모델을 준비하고 입력 관절각을 그대로 반영한다.
모델 XML과 모터 설정을 수정하거나 물리 시간을 진행하지 않는다.
"""

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from kinematics.anchoring import TipAnchor
from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES, as_joint_angles
from kinematics.poses import invert_pose
from planning.motion_path import MotionPath


def set_arm_angles(model: mujoco.MjModel, data: mujoco.MjData, q_rad: ArrayLike) -> None:
    """J1부터 J7까지의 입력 각도를 자르지 않고 적용한 뒤 기구학 상태를 갱신한다."""
    angles = as_joint_angles(q_rad)
    for name, angle in zip(ARM_JOINT_NAMES, angles, strict=True):
        data.joint(name).qpos[0] = angle
    mujoco.mj_forward(model, data)


def apply_anchor(model: mujoco.MjModel, data: mujoco.MjData, anchor: TipAnchor) -> None:
    r"""현재 관절각을 유지하며 지정된 팁이 고정 자세에 놓이도록 로봇 뿌리를 옮긴다.

    $$
    T_{B,\mathrm{new}}=T_{F,\mathrm{target}}T_{F,\mathrm{current}}^{-1}T_{B,\mathrm{current}}
    $$

    모든 자세는 월드 기준이며 B는 로봇 뿌리, F는 고정 팁이다.
    호출 전 관절각과 mj_forward 상태가 갱신되어 있어야 한다. 빔 배치는 유지한다.
    """
    fixed_site = data.site(anchor.fixed_tip)
    fixed_pose = np.eye(4)
    fixed_pose[:3, :3] = fixed_site.xmat.reshape(3, 3)
    fixed_pose[:3, 3] = fixed_site.xpos
    root = model.body("gripper_L")
    root_pose = np.eye(4)
    root_pose[:3, :3] = data.body("gripper_L").xmat.reshape(3, 3)
    root_pose[:3, 3] = data.body("gripper_L").xpos
    # 현재 고정 팁을 목표 고정 자세로 옮기는 변환: $$\Delta T=T_{F,\mathrm{target}}T_{F,\mathrm{current}}^{-1}$$
    correction = anchor.T_world_fixed_tip @ invert_pose(fixed_pose)
    # 같은 변환을 로봇 뿌리에 적용: $$T_{B,\mathrm{new}}=\Delta T T_{B,\mathrm{current}}$$
    placed_root = correction @ root_pose
    root.pos[:] = placed_root[:3, 3]
    mujoco.mju_mat2Quat(root.quat, placed_root[:3, :3].reshape(-1))
    mujoco.mj_forward(model, data)


def set_path_time(
    model: mujoco.MjModel, data: mujoco.MjData, motion_path: MotionPath, elapsed_s: float,
) -> int:
    """경로의 지정 시각을 모델에 적용하고 현재 구간 번호를 반환한다."""
    segment_index, q_rad, gripper_q_rad = motion_path.sample(elapsed_s)
    for index, name in enumerate(GRIPPER_JOINT_NAMES):
        address = model.jnt_qposadr[model.joint(name).id]
        data.joint(name).qpos[0] = model.qpos0[address] if gripper_q_rad is None else gripper_q_rad[index]
    set_arm_angles(model, data, q_rad)
    apply_anchor(model, data, motion_path.segments[segment_index].anchor)
    return segment_index


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
