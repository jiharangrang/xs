"""원본 CAD의 고정 그리퍼와 빔 사이 이론 파지 관계를 검사한다."""

import mujoco
import numpy as np

from common import MODEL


def original_beam_pose():
    """카메라 관측 없이 원본 XML의 빔 배치만 반환한다."""
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    return {"pos": model.body("ibeam").pos.tolist(), "quat": model.body("ibeam").quat.tolist(),
            "source": "original_cad_fixed_left_grasp", "ideal_rigid_scene": True}


def support_geometry(model, data):
    r"""고정 팁과 빔 하부 플랜지 안쪽 접촉면의 간격과 기울기를 계산한다.

    $$g=p_{L,B,y}-y_{inner},\qquad p_{L,B}=R_{WB}^{T}(p_{WL}-p_{WB})$$

    하부 플랜지의 바깥면과 안쪽면을 구분하며 양쪽 면 사이의 두께를 간격으로 오인하지 않는다.
    """
    mujoco.mj_kinematics(model, data)
    geom = model.geom("ibeam_mesh").id
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    # 메시 꼭짓점의 월드 좌표: $$v_W=R_{WG}v_G+p_{WG}$$
    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    beam = data.body("ibeam")
    rotation = beam.xmat.reshape(3, 3)
    # 꼭짓점을 빔 기준으로 변환: $$v_B=R_{WB}^{T}(v_W-p_{WB})$$
    local = (world - beam.xpos) @ rotation
    lower, upper = local.min(axis=0), local.max(axis=0)
    levels = np.unique(np.round(local[:, 1], 7))
    inner_y = levels[-2]
    tip = data.site("tip_L")
    # 고정 팁의 빔 좌표: $$p_{L,B}=R_{WB}^{T}(p_{WL}-p_{WB})$$
    point = rotation.T @ (tip.xpos - beam.xpos)
    # 하부 플랜지 안쪽 접촉면 간격: $$g=p_{L,B,y}-y_{inner}$$
    gap = point[1] - inner_y
    # 두 접촉면 법선 사이 기울기: $$\alpha=\arccos(|n_L^T n_B|)$$
    angle = np.arccos(np.clip(abs(tip.xmat.reshape(3, 3)[:, 2] @ rotation[:, 1]), 0, 1))
    return {"inner_plane_gap_m": float(gap), "normal_error_rad": float(angle),
            "inside_flange_footprint": bool(lower[0] < point[0] < upper[0] and lower[2] < point[2] < upper[2]),
            "fixed_jaw_rad": float(data.joint("G_L").qpos[0]),
            "scope": "CAD geometry and rigid anchor; not contact-force or friction validation"}


def require_initial_grasp(model, data):
    """원본 고정 그리퍼의 파지 관계가 어긋난 장면은 학습 전에 거부한다."""
    result = support_geometry(model, data)
    if (abs(result["inner_plane_gap_m"]) > 1e-6 or result["normal_error_rad"] > 1e-5
            or not result["inside_flange_footprint"] or abs(result["fixed_jaw_rad"]) > 1e-7):
        raise ValueError(f"원본 CAD의 왼쪽 고정 파지 상태가 깨졌습니다: {result}")
    return result


def check_ideal_path(scene, q):
    r"""구간의 사분점까지 검사하여 고정 파지 유지와 오른쪽 그리퍼의 비관통을 확인한다.

    $$q_i(u)=q_i+u(q_{i+1}-q_i),\qquad u\in\{0,1/4,1/2,3/4\}$$

    유한한 표본 검사이며 연속 경로 전체의 충돌 부재 증명은 아니다.
    """
    # 각 관절 보간 구간의 사분점: $$q_i(u)=q_i+u(q_{i+1}-q_i)$$
    segments = [q[i] + np.arange(4)[:, None] / 4 * (q[i + 1] - q[i]) for i in range(len(q) - 1)]
    dense = np.concatenate(segments + [q[-1:]])
    distances = scene.label(dense)
    for sample in dense:
        scene.set_q(sample)
        require_initial_grasp(scene.model, scene.data)
    report = {"samples": len(dense), "minimum_distance_m": float(distances.min()),
              "penetrating_samples": int(np.sum(distances < 0)), "support_valid_all_samples": True,
              "continuous_collision_proof": False, "scope": scene.config["scope"]}
    if report["penetrating_samples"]:
        raise ValueError(f"이론 경로에서 오른쪽 그리퍼가 빔을 관통합니다: {report}")
    return dense, distances, report
