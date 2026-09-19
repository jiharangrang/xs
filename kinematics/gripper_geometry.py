"""실측 관절각에서 움직이는 그리퍼 형상을 깊이 카메라 좌표로 표현한다.
고정턱의 상단 돌출부와 열린 턱을 구분해 높이·횡방향 간격 계산에 제공한다.
"""

import mujoco
import numpy as np

from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import ARM_JOINT_NAMES, as_joint_angles
from kinematics.mesh_sections import clip_triangles


class GripperGeometry:
    """공통 CAD 메시와 카메라 장착 좌표를 사용하며 실물 명령을 내리지 않는다."""

    def __init__(self):
        r"""고정턱 접촉 높이 이상의 돌출부를 메시에서 선택한다.

        $$p_B=R_{WB}^T(p_W-p_{WB})$$
        """
        self.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._meshes = {}
        self._faces = {}
        for name in ("gripper_R_geom_0", "gripper_R_thumb_geom_0"):
            geom = self.model.geom(name).id
            mesh = self.model.geom_dataid[geom]
            begin = self.model.mesh_vertadr[mesh]
            self._meshes[name] = (geom, self.model.mesh_vert[begin:begin + self.model.mesh_vertnum[mesh]].copy())
            begin_face = self.model.mesh_faceadr[mesh]
            self._faces[name] = self.model.mesh_face[begin_face:begin_face + self.model.mesh_facenum[mesh]].copy()
        fixed = self._world_points("gripper_R_geom_0")
        body = self.data.body("gripper_R")
        # 고정턱 꼭짓점을 그리퍼 몸체 좌표로 표현: $$p_B=R_{WB}^T(p_W-p_{WB})$$
        local = (fixed - body.xpos) @ body.xmat.reshape(3, 3)
        height = self.model.site("tip_R").pos[2]
        self._lip = local[:, 2] >= height - 1e-6
        if np.count_nonzero(self._lip) < 4:
            raise ValueError("모델에서 고정턱의 상단 돌출부를 찾지 못했습니다.")
        self._lower_fixed = clip_triangles(local[self._faces["gripper_R_geom_0"]],
                                           np.array([0., 0., 1.]), height - 1e-7)

    def update(self, q_rad, grippers_deg):
        r"""실측 몸통·그리퍼 관절각을 모델에 반영한다.

        $$q_{grip}=q_{deg}\pi/180$$
        """
        self.data.qpos[:] = self.model.qpos0
        for name, angle in zip(ARM_JOINT_NAMES, as_joint_angles(q_rad), strict=True):
            self.data.joint(name).qpos[0] = angle
        for name in ("G_L", "G_R"):
            angle = grippers_deg[name]
            if not np.isfinite(angle):
                raise ValueError("그리퍼의 실제 각도가 필요합니다.")
            # 실측 그리퍼 각도를 라디안으로 변환: $$q_{grip}=q_{deg}\pi/180$$
            self.data.joint(name).qpos[0] = np.deg2rad(angle)
        mujoco.mj_forward(self.model, self.data)

    def _world_points(self, name):
        r"""지정한 메시의 꼭짓점을 월드 좌표로 변환한다.

        $$p_W=R_{WG}p_G+p_{WG}$$
        """
        geom, vertices = self._meshes[name]
        # 메시 좌표를 월드 좌표로 변환: $$p_W=R_{WG}p_G+p_{WG}$$
        return vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]

    def points_camera(self, *, fixed_lip=False):
        r"""전체 그리퍼 또는 고정턱 돌출부의 꼭짓점을 깊이 카메라 좌표로 반환한다.

        $$p_C=R_{WC}^T(p_W-p_{WC})$$

        돌출부의 최소 횡좌표는 사용자가 지정한 안쪽 옆면과 둥근 모서리를 포함한다.
        """
        fixed = self._world_points("gripper_R_geom_0")
        world = fixed[self._lip] if fixed_lip else np.vstack((fixed, self._world_points("gripper_R_thumb_geom_0")))
        camera = self.data.site("depth_frame")
        # 월드 꼭짓점을 카메라 좌표로 변환: $$p_C=R_{WC}^T(p_W-p_{WC})$$
        return (world - camera.xpos) @ camera.xmat.reshape(3, 3)

    def outward_camera(self):
        r"""고정턱 안쪽 면에서 바깥쪽으로 향하는 몸체 축을 카메라 좌표로 반환한다.

        $$u_C=R_{WC}^TR_{WB}[0,1,0]^T$$
        """
        # 고정턱이 놓인 양의 몸체 y축을 카메라로 변환: $$u_C=R_{WC}^TR_{WB}e_y$$
        return self.data.site("depth_frame").xmat.reshape(3, 3).T @ self.data.body("gripper_R").xmat.reshape(3, 3)[:, 1]

    def lower_triangles_camera(self):
        r"""고정턱 접촉면 아래의 몸체와 가동턱 전체를 카메라 좌표로 반환한다.

        $$p_C=R_{WC}^T(R_{WB}p_B+p_{WB}-p_{WC})$$
        """
        body = self.data.body("gripper_R")
        # 잘라 둔 고정턱 아래쪽 메시를 월드 좌표로 변환: $$p_W=R_{WB}p_B+p_{WB}$$
        fixed = self._lower_fixed @ body.xmat.reshape(3, 3).T + body.xpos
        thumb = self._world_points("gripper_R_thumb_geom_0")[self._faces["gripper_R_thumb_geom_0"]]
        world = np.concatenate((fixed, thumb), axis=0)
        camera = self.data.site("depth_frame")
        # 고정턱 아래쪽과 가동턱을 카메라로 변환: $$p_C=R_{WC}^T(p_W-p_{WC})$$
        return (world - camera.xpos) @ camera.xmat.reshape(3, 3)

    def rgb_center_on_plane(self, normal, plane_offset_m):
        r"""RGB 광축이 관측 빔 평면과 만나는 점을 깊이 카메라 좌표로 반환한다.

        $$p_c=o+t v,\quad t=-\frac{n^To+d}{n^Tv}$$

        o와 v는 깊이 좌표로 옮긴 RGB 센서 원점과 광축이며 두 센서의 장착 차이를 반영한다.
        """
        depth = self.data.site("depth_frame")
        rgb = self.data.site("rgb_frame")
        rotation = depth.xmat.reshape(3, 3)
        # RGB 원점을 깊이 좌표로 변환: $$o=R_{WD}^T(p_{WR}-p_{WD})$$
        origin = rotation.T @ (rgb.xpos - depth.xpos)
        # RGB 광축을 깊이 좌표로 변환: $$v=R_{WD}^TR_{WR}e_z$$
        axis = rotation.T @ rgb.xmat.reshape(3, 3)[:, 2]
        # 광축의 평면 법선 방향 성분: $$a=n^Tv$$
        denominator = float(normal @ axis)
        if not np.isfinite(denominator) or denominator >= -1e-8:
            raise ValueError("RGB 중앙 광축 앞에서 빔 평면을 확인할 수 없습니다.")
        # 광축을 따라 평면까지의 거리: $$t=-(n^To+d)/a$$
        distance = -float(normal @ origin + plane_offset_m) / denominator
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("RGB 센서 앞의 유효한 빔 거리가 필요합니다.")
        # RGB 중앙이 가리키는 빔 평면 위 점: $$p_c=o+tv$$
        return origin + distance * axis
