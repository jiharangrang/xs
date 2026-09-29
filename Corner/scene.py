"""기존 CAD를 읽어 수평 ㄱ자 빔과 로봇의 독립적인 기구학 장면을 구성한다.
삼각형 메시 거리 검사를 사용하며 장치 통신이나 접촉 동역학은 실행하지 않는다.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import fcl
import mujoco
import numpy as np
import trimesh

from kinematics.anchoring import TipAnchor
from kinematics.joints import ARM_JOINT_NAMES
from simulation.model import apply_anchor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "models/xs/model.xml"
OUTPUT = ROOT / "outputs"


def mesh_arrays(model, geom):
    """MuJoCo 메시의 정점과 삼각형을 복사해 반환한다."""
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    face = model.mesh_faceadr[mesh]
    return (model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(float),
            model.mesh_face[face:face + model.mesh_facenum[mesh]].copy())


def world_vertices(model, data, geom):
    r"""메시 정점을 현재 월드 좌표로 옮긴다.

    $$v_W=v_G R_{WG}^{T}+p_{WG}$$
    """
    vertices, _ = mesh_arrays(model, geom)
    # 행 벡터 정점의 월드 변환: $$v_W=v_G R_{WG}^{T}+p_{WG}$$
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def site_pose(data, name):
    """사이트 위치와 방향을 동차변환 행렬로 반환한다."""
    pose = np.eye(4)
    pose[:3, :3] = data.site(name).xmat.reshape(3, 3)
    pose[:3, 3] = data.site(name).xpos
    return pose


def build_scene(corner_x=.16, directory=OUTPUT / "scene"):
    r"""원본 I빔 단면을 보존한 두 빔을 사선 접합해 수평 직각 장면을 저장한다.

    $$x_{A,end}=c-u,\qquad (x_B,y_{B,start})=(c-u,u)$$

    c는 모서리 중심의 x좌표, u는 빔 중심에서 측정한 단면의 횡좌표다.
    두 빔의 끝단은 같은 사선 평면을 공유한다.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = model.geom("ibeam_mesh").id
    vertices = world_vertices(model, data, geom)
    _, faces = mesh_arrays(model, geom)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    # 원본 빔 길이 방향의 정규화 좌표: $$t=(x-x_{min})/(x_{max}-x_{min})$$
    fraction = (vertices[:, 0] - lower[0]) / (upper[0] - lower[0])
    # 빔 단면 중앙 기준 횡좌표: $$u=y-(y_{min}+y_{max})/2$$
    lateral = vertices[:, 1] - (lower[1] + upper[1]) / 2
    beams = []
    for name in ("A", "B"):
        transformed = vertices.copy()
        if name == "A":
            # 직진 빔의 사선 끝단까지 보간: $$x_A=-0.4+t(c-u+0.4)$$
            transformed[:, 0] = -.4 + fraction * (corner_x - lateral + .4)
            transformed[:, 1] = lateral
        else:
            # 회전 빔의 폭 방향 좌표: $$x_B=c-u$$
            transformed[:, 0] = corner_x - lateral
            # 같은 사선 접합부에서 출발: $$y_B=u+t(0.5-u)$$
            transformed[:, 1] = lateral + fraction * (.5 - lateral)
        path = directory / f"beam_{name}.stl"
        trimesh.Trimesh(transformed, faces, process=False).export(path)
        beams.append(path)
    xml = ET.parse(SOURCE).getroot()
    for asset in xml.findall("./asset/mesh"):
        asset.set("file", str(SOURCE.parent / asset.attrib["file"]))
    for include in xml.findall(".//include"):
        include.set("file", str(SOURCE.parent / include.attrib["file"]))
    world = xml.find("worldbody")
    world.remove(world.find("body[@name='ibeam']"))
    for name, path, color in zip(("A", "B"), beams, ("0.42 0.54 0.66 1", "0.30 0.64 0.61 1")):
        ET.SubElement(xml.find("asset"), "mesh", name=f"beam_{name}", file=str(path))
        ET.SubElement(world, "geom", name=f"beam_{name}", type="mesh", mesh=f"beam_{name}",
                      rgba=color, contype="0", conaffinity="0")
    ET.SubElement(xml.find("asset"), "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.91 0.94 0.97", rgb2="0.98 0.99 1", width="256", height="256")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", pos="0 0 -0.14",
                  size="2 2 .01", rgba="0.94 0.95 0.96 1", contype="0", conaffinity="0")
    for geom in xml.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    path = directory / "model.xml"
    ET.ElementTree(xml).write(path, encoding="unicode")
    return path


class Scene:
    """원본 CAD 메시를 유지한 로봇 배치와 충돌 질의를 제공한다."""

    def __init__(self, path):
        """독립 모델과 삼각형 BVH를 준비한다."""
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = 1600
        self.model.vis.global_.offheight = 1000
        self.model.vis.headlight.ambient[:] = [.55, .55, .55]
        self.model.vis.headlight.diffuse[:] = [.7, .7, .7]
        self.model.vis.headlight.specular[:] = [.1, .1, .1]
        for geom in range(self.model.ngeom):
            if self.model.geom(geom).name not in ("beam_A", "beam_B", "floor"):
                if np.max(self.model.geom_rgba[geom, :3]) < .15:
                    self.model.geom_rgba[geom, :3] = [.16, .19, .24]
        self.arm = [self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM_JOINT_NAMES]
        self.grip = [self.model.jnt_qposadr[self.model.joint(n).id] for n in ("G_L", "G_R")]
        self.root_pos = self.model.body("gripper_L").pos.copy()
        self.root_quat = self.model.body("gripper_L").quat.copy()
        self.objects = {}
        self.names = {}
        for geom in range(self.model.ngeom):
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            vertices, faces = mesh_arrays(self.model, geom)
            bvh = fcl.BVHModel()
            bvh.beginModel(len(vertices), len(faces))
            bvh.addSubModel(vertices, faces)
            bvh.endModel()
            self.objects[geom] = fcl.CollisionObject(bvh)
            self.names[geom] = self.model.geom(geom).name
        self.beams = [g for g, n in self.names.items() if n.startswith("beam_")]
        self.robot = [g for g in self.names if g not in self.beams]
        self.distance_request = fcl.DistanceRequest(enable_nearest_points=False)
        self.collision_request = fcl.CollisionRequest(num_max_contacts=1)

    def set(self, q, grippers, anchor=None):
        """관절값과 고정 팁을 적용하고 모든 충돌 메시의 강체 배치를 갱신한다."""
        self.model.body("gripper_L").pos[:] = self.root_pos
        self.model.body("gripper_L").quat[:] = self.root_quat
        self.data.qpos[self.arm] = q
        self.data.qpos[self.grip] = grippers
        mujoco.mj_kinematics(self.model, self.data)
        if anchor is not None:
            apply_anchor(self.model, self.data, anchor)
        for geom, obj in self.objects.items():
            obj.setTransform(fcl.Transform(self.data.geom_xmat[geom].reshape(3, 3), self.data.geom_xpos[geom]))

    def beam_distances(self):
        """각 로봇 메시와 두 빔 사이의 최단거리와 교차 여부를 반환한다."""
        records = []
        for robot in self.robot:
            for beam in self.beams:
                a, b = self.objects[robot], self.objects[beam]
                collision = fcl.collide(a, b, self.collision_request, fcl.CollisionResult())
                distance = fcl.distance(a, b, self.distance_request, fcl.DistanceResult())
                records.append((self.names[robot], self.names[beam], float(distance), bool(collision)))
        return records

    def self_collisions(self):
        """같은 몸체와 직접 연결된 몸체를 제외한 로봇 메시 교차를 반환한다."""
        result = []
        for index, first in enumerate(self.robot):
            body_a = self.model.geom_bodyid[first]
            for second in self.robot[index + 1:]:
                body_b = self.model.geom_bodyid[second]
                if (body_a == body_b or self.model.body_parentid[body_a] == body_b
                        or self.model.body_parentid[body_b] == body_a):
                    continue
                if fcl.collide(self.objects[first], self.objects[second], self.collision_request,
                               fcl.CollisionResult()):
                    result.append((self.names[first], self.names[second]))
        return result
