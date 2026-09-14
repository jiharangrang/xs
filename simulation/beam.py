"""현재 I빔 CAD 형상에서 깊이 카메라 기준의 하부면 정답을 계산한다.
관측 알고리즘의 검증용이며 실물에서 빔을 검출하는 데 사용하지 않는다.
"""

import mujoco
import numpy as np


def beam_reference(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    r"""CAD 하부면의 법선·길이 방향·폭·중심선을 카메라 좌표로 옮긴다.

    $$p_C=R_{WC}^T(p_W-p_{WC}),\quad v_C=R_{WC}^Tv_W$$

    W는 월드, C는 깊이 광학 좌표다. 현재 빔 몸체의 x는 길이, y는 하부면 바깥쪽,
    z는 폭 방향이며, 메시 꼭짓점으로 실제 폭과 하부면 높이를 구한다.
    """
    geom = model.geom("ibeam_mesh").id
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    # 메시 점을 월드 좌표로 변환: $$p_W=R_{WG}p_G+p_{WG}$$
    world_vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    body = data.body("ibeam")
    rotation_wb = body.xmat.reshape(3, 3)
    # 메시 점을 빔 몸체 기준으로 변환: $$p_B=R_{WB}^T(p_W-p_{WB})$$
    local_vertices = (world_vertices - body.xpos) @ rotation_wb
    lower, upper = local_vertices.min(axis=0), local_vertices.max(axis=0)
    midpoint = (lower + upper) / 2
    midpoint[1] = upper[1]
    # 하부면 중심을 월드 좌표로 이동: $$p_W=R_{WB}p_B+p_{WB}$$
    point_world = rotation_wb @ midpoint + body.xpos
    camera = data.camera("gemini215_depth")
    # 렌더링 카메라 좌표축을 광학 좌표축으로 변환: $$R_{WC}=R_{W,GL}\operatorname{diag}(1,-1,-1)$$
    rotation_wc = camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1])
    # 하부면 점의 깊이 카메라 좌표: $$p_C=R_{WC}^T(p_W-p_{WC})$$
    point = rotation_wc.T @ (point_world - camera.xpos)
    # 빔 몸체 y축인 하부면 법선의 카메라 좌표: $$n_C=R_{WC}^TR_{WB}e_y$$
    normal = rotation_wc.T @ rotation_wb[:, 1]
    if normal @ point > 0:
        normal = -normal
    # 빔 몸체 x축인 길이 방향의 카메라 좌표: $$t_C=R_{WC}^TR_{WB}e_x$$
    axis = rotation_wc.T @ rotation_wb[:, 0]
    # 카메라에 가장 가까운 중심선 점: $$c=p_C-t_C(t_C^Tp_C)$$
    centerline = point - axis * (axis @ point)
    # 카메라 기준 평면 상수: $$d=-n_C^Tp_C$$
    offset = -float(normal @ point)
    # 몸체 z축 방향의 메시 전체 폭: $$w=z_{max}-z_{min}$$
    width = float(upper[2] - lower[2])
    return {"coordinate_frame": "depth_camera_optical", "normal": normal.tolist(),
            "plane_offset_m": offset, "axis": axis.tolist(), "width_m": width,
            "centerline_point_m": centerline.tolist()}
