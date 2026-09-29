"""유격·처짐·실측 보정 없이 원본 CAD의 고정 파지 상태에서 이론 6단계 경로를 만든다.
기존 IK와 직선 이동 계산만 재사용하며 카메라·모터 기록은 읽지 않는다.
"""

import json
import sys

import mujoco
import numpy as np

from common import JOINTS, MODEL, REPO, file_hash
from support import require_initial_grasp

sys.path.insert(0, str(REPO))
from kinematics.anchoring import TipAnchor
from kinematics.ik import IKSettings, InverseKinematics
from planning.linear_motion import plan_linear_motion
from planning.targets import beam_grasp_target

REFERENCE = REPO / "outputs/stage1_observation_pose.json"


def world_vertices(model, data, name):
    r"""지정한 원본 메시의 꼭짓점을 월드 좌표로 읽는다.

    $$v_W=R_{WG}v_G+p_{WG}$$
    """
    geom = model.geom(name).id
    mesh = model.geom_dataid[geom]
    begin = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[begin:begin + model.mesh_vertnum[mesh]]
    # 메시의 월드 꼭짓점: $$v_W=R_{WG}v_G+p_{WG}$$
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def ideal_trace(reference=REFERENCE, dt=.1, *, solver=None, segment_guard=None, recompute_reference=False):
    r"""왼쪽 파지를 고정한 이론 경로를 관측·정렬·상승·이탈·상승·삽입 순서로 만든다.

    $$q_1(u)=q_0+(10u^3-15u^4+6u^5)(q_{obs}-q_0)$$

    첫 단계는 관절 보간, 이후 이동 단계는 고정 팁을 보존한 직선 IK다.
    이론상 정렬 오차가 없으므로 두 번째 단계는 같은 자세를 유지한다.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("재생 간격은 유한한 양수여야 합니다.")
    reference = reference.resolve()
    saved = json.loads(reference.read_text())
    if saved.get("format") != "xs.observation_pose.v1" or saved.get("fixed_tip") != "tip_L":
        raise ValueError("왼쪽 고정 이론 관측 자세가 필요합니다.")
    solver = InverseKinematics(settings=IKSettings(starts=1)) if solver is None else solver
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    support = require_initial_grasp(model, data)
    q_initial = np.zeros(7)
    q_observation = np.asarray(saved["q_rad"], dtype=float)
    q_grasp = np.asarray(saved["q_grasp_rad"], dtype=float)
    if recompute_reference:
        reference_solver = InverseKinematics(settings=IKSettings(starts=1))
        initial = reference_solver.fk.forward(q_initial)
        reference_anchor = TipAnchor("tip_L", initial.T_world_tip_L)
        # 서로 마주 보는 두 팁의 기존 파지 목표를 월드로 변환: $${}^W T_{R,d}={}^W T_L{}^L T_{R,d}$$
        reference_goal = initial.T_world_tip_L @ beam_grasp_target(.2)
        result = reference_solver.solve(reference_anchor.to_relative_target(reference_goal), np.round(q_grasp, 1))
        if not result.success:
            raise RuntimeError("20 cm 앞 기준 자세의 IK를 다시 계산하지 못했습니다.")
        q_grasp = result.q_rad
        reference_goal[2, 3] -= .05
        result = reference_solver.solve(reference_anchor.to_relative_target(reference_goal), q_grasp)
        if not result.success:
            raise RuntimeError("관측 자세의 IK를 다시 계산하지 못했습니다.")
        q_observation = result.q_rad
    grasp = solver.fk.forward(q_grasp)
    anchor = TipAnchor("tip_L", grasp.T_world_tip_L)
    goal = grasp.T_world_tip_R.copy()
    if not np.allclose(goal[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("수평 빔에 정렬된 이론 파지 목표가 아닙니다.")
    u = np.linspace(0, 1, 61)
    # 시작과 끝의 속도를 영으로 하는 진행률: $$s(u)=10u^3-15u^4+6u^5$$
    progress = 10 * u**3 - 15 * u**4 + 6 * u**5
    # 관측 시작 자세까지의 이상적인 관절 보간: $$q(u)=q_0+s(u)(q_{obs}-q_0)$$
    first = q_initial + progress[:, None] * (q_observation - q_initial)
    if segment_guard is not None:
        segment_guard(first)
    segments = [(1, "observation_pose", first), (2, "already_aligned", np.repeat(q_observation[None], 10, axis=0))]
    current = q_observation.copy()
    open_angle = np.deg2rad(-120.)
    for name, value in zip(JOINTS, np.r_[current, open_angle], strict=True):
        data.joint(name).qpos[0] = value
    mujoco.mj_kinematics(model, data)
    beam = world_vertices(model, data, "ibeam_mesh")
    fixed = world_vertices(model, data, "gripper_R_geom_0")
    thumb = world_vertices(model, data, "gripper_R_thumb_geom_0")
    tip = data.site("tip_R").xpos.copy()
    # 팁보다 높은 그리퍼 꼭짓점의 높이: $$h_{top}=\max_i z_i-z_{tip}$$
    top_offset = max(fixed[:, 2].max(), thumb[:, 2].max()) - tip[2]
    # 빔 아래 목표 간격을 유지할 팁 높이: $$z_3=z_{beam,bottom}-0.025-h_{top}$$
    lift_z = beam[:, 2].min() - .025 - top_offset
    lip = fixed[fixed[:, 2] >= tip[2] - 1e-6]
    # 고정턱 돌출부와 빔 옆면의 목표 여유: $$\Delta y=y_{beam,max}+0.010-y_{lip,min}$$
    lateral = beam[:, 1].max() + .010 - lip[:, 1].min()
    # 삽입 전 안쪽 접촉면에서 확보할 여유: $$z_5=z_{grasp}+0.001$$
    insertion_z = goal[2, 3] + .001
    position = solver.fk.forward(current).T_world_tip_R[:3, 3].copy()
    targets = [(3, "lift_below_beam", np.array([position[0], position[1], lift_z])),
               (4, "side_exit", np.array([position[0], position[1] + lateral, lift_z])),
               (5, "lift_to_insertion_height", np.array([position[0], position[1] + lateral, insertion_z])),
               (6, "side_insert", np.array([position[0], position[1], insertion_z]))]
    for stage, name, target in targets:
        previous = solver.fk.forward(current).T_world_tip_R[:3, 3]
        # 직선 IK에 전달할 목표 변위: $$\Delta p=p_{target}-p_{current}$$
        displacement = target - previous
        steps = max(2, int(np.ceil(np.linalg.norm(displacement) / .001)))
        result = plan_linear_motion(current, anchor, displacement, steps=steps, solver=solver)
        if not result.success:
            raise RuntimeError(f"이론 {stage}단계 IK 실패: {result.message}")
        segments.append((stage, name, result.q_path_rad[1:]))
        current = result.q_path_rad[-1]
    arm = np.concatenate([segment[2] for segment in segments])
    q = np.column_stack((arm, np.full(len(arm), open_angle)))
    stages = np.concatenate([np.full(len(angles), stage) for stage, _, angles in segments])
    times = np.arange(len(q)) * dt
    metadata = {"source": "ideal_cad_kinematic_stages", "measured_logs_used": False,
                "reference": str(reference), "reference_sha256": file_hash(reference),
                "joint_names": list(JOINTS), "fixed_tip": "tip_L", "fixed_jaw_rad": 0.,
                "initial_q_rad": q[0].tolist(), "support": support,
                "stages": [{"stage": stage, "name": name, "samples": len(angles)} for stage, name, angles in segments],
                "dt_s": dt, "below_gap_m": .025, "side_gap_m": .010, "insertion_gap_m": .001,
                "reference_poses_recomputed": recompute_reference,
                "interpretation": "이론 기구학 경로; 6단계는 열린 오른쪽 그리퍼의 삽입 완료이며 닫기와 동역학 검증은 제외"}
    return q, times, stages, metadata
