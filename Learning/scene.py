"""고정된 학습 장면에서 움직이는 오른쪽 그리퍼와 빔의 메시 충돌거리를 계산한다.
볼록 조각으로 근사한 CAD 메시의 부호 있는 최소 쌍별 거리를 정답으로 사용한다.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common import ARTIFACTS, GEOMS, JOINTS, MODEL, file_hash, signature, source_mesh_path
from support import require_initial_grasp


def checked_q(q, limits):
    """유한한 관절 배열과 학습 장면의 관절 범위를 검사한다."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (len(JOINTS),) or not np.all(np.isfinite(q)):
        raise ValueError("유한한 관절각 여덟 개가 필요합니다.")
    if np.any(q < limits[:, 0] - 1e-7) or np.any(q > limits[:, 1] + 1e-7):
        raise ValueError("학습 장면의 관절 범위를 벗어났습니다.")
    return q


class DistanceScene:
    """기하 거리 정답과 학습 입력이 공유하는 불변 장면을 읽는다."""

    def __init__(self, directory=ARTIFACTS / "scene"):
        """원본·기하 식별자를 확인하고 독립적인 MuJoCo 모델과 데이터를 만든다."""
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "scene.json").read_text())
        expected = self.config["scene_id"]
        if signature({k: v for k, v in self.config.items() if k != "scene_id"}) != expected:
            raise ValueError("장면 설정 식별자가 일치하지 않습니다.")
        for path, digest in ((MODEL, self.config["model_sha256"]),
                             (MODEL.parent / "camera_frames.xml", self.config["camera_frames_sha256"]),
                             (MODEL.parent / "calibration.yaml", self.config["calibration_sha256"]),
                             (self.directory / "geometry.npz", self.config["geometry_sha256"])):
            if file_hash(path) != digest:
                raise ValueError(f"장면 생성 이후 파일이 바뀌었습니다: {path}")
        for name, info in self.config["parts"].items():
            if file_hash(source_mesh_path(name)) != info["sha256"]:
                raise ValueError("원본 충돌 메시가 변경되었습니다.")
        xml = ET.parse(MODEL).getroot()
        for asset in xml.findall("./asset/mesh"):
            asset.set("file", str(MODEL.parent / asset.attrib["file"]))
        for include in xml.findall(".//include"):
            include.set("file", str(MODEL.parent / include.attrib["file"]))
        for geom in xml.findall(".//geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        beam = xml.find(".//body[@name='ibeam']")
        for key in ("pos", "quat"):
            beam.set(key, " ".join(map(str, self.config["beam_pose"][key])))
        owners = {"gripper_R_geom_0": "gripper_R", "gripper_R_thumb_geom_0": "gripper_R_thumb", "ibeam_mesh": "ibeam"}
        with np.load(self.directory / "geometry.npz", allow_pickle=False) as arrays:
            for name in GEOMS:
                body = xml.find(f".//body[@name='{owners[name]}']")
                for i in range(self.config["parts"][name]["count"]):
                    key = f"{name}_{i}"
                    ET.SubElement(xml.find("asset"), "mesh", name=key,
                                  vertex=" ".join(map(str, arrays[key + "_v"].ravel())),
                                  face=" ".join(map(str, arrays[key + "_f"].ravel())))
                    ET.SubElement(body, "geom", name=key, type="mesh", mesh=key, mass="0",
                                  contype="0", conaffinity="0", group="3", rgba="0 1 0 0.2")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        if self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD):
            raise ValueError("부호 있는 거리 정답에는 native CCD가 필요합니다.")
        self.data = mujoco.MjData(self.model)
        self.support = require_initial_grasp(self.model, self.data)
        self.limits = np.asarray(self.config["limits"], dtype=float)
        self.qpos = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINTS])
        self.robot_geoms = [self.model.geom(f"{name}_{i}").id for name in GEOMS[:2]
                            for i in range(self.config["parts"][name]["count"])]
        self.beam_geoms = [self.model.geom(f"ibeam_mesh_{i}").id for i in range(self.config["parts"]["ibeam_mesh"]["count"])]
        self.scene_id = expected
        self.pair_a = np.repeat(self.robot_geoms, len(self.beam_geoms))
        self.pair_b = np.tile(self.beam_geoms, len(self.robot_geoms))
        self.last_narrow_queries = 0

    def set_q(self, q):
        """접촉 동역학을 진행하지 않고 관절각의 기하 배치만 갱신한다."""
        self.data.qpos[self.qpos] = checked_q(q, self.limits)
        mujoco.mj_kinematics(self.model, self.data)

    def distance(self, q):
        r"""오른쪽 그리퍼와 빔의 모든 볼록 조각 쌍에서 부호 있는 최소 거리를 반환한다.

        $$d(q)=\min_{(i,j)\in\mathcal P}d_{ij}(q)$$

        P는 움직이는 오른쪽 고정턱·가동턱과 빔 사이의 조각 쌍이다.
        지지 중인 왼쪽 그리퍼 접촉과 다른 링크·자기충돌은 이 출력의 범위 밖이다.
        """
        self.set_q(q)
        # 조각 중심 간 거리: $$r_{ij}=\|c_i-c_j\|_2$$
        separation = np.linalg.norm(self.data.geom_xpos[self.pair_a] - self.data.geom_xpos[self.pair_b], axis=1)
        # 두 외접 구 사이 분리 거리의 하한: $$\ell_{ij}=r_{ij}-r_i-r_j$$
        lower = separation - self.model.geom_rbound[self.pair_a] - self.model.geom_rbound[self.pair_b]
        rotation = self.data.geom_xmat.reshape(-1, 3, 3)
        # 각 조각의 월드 축 정렬 상자 중심: $$c_W=R c_G+p_W$$
        centers = np.einsum("nij,nj->ni", rotation, self.model.geom_aabb[:, :3]) + self.data.geom_xpos
        # 회전된 상자를 감싸는 축별 반폭: $$h_W=|R|h_G$$
        half = np.einsum("nij,nj->ni", np.abs(rotation), self.model.geom_aabb[:, 3:])
        # 축별 상자 분리 간격: $$g=\max(|c_i-c_j|-h_i-h_j,0)$$
        gap = np.maximum(np.abs(centers[self.pair_a] - centers[self.pair_b]) - half[self.pair_a] - half[self.pair_b], 0)
        # 분리된 상자의 거리로 외접 구 하한을 보강: $$\ell=\max(\ell_{sphere},\|g\|_2)$$
        lower = np.maximum(lower, np.linalg.norm(gap, axis=1))
        best = 10.
        self.last_narrow_queries = 0
        for index in np.argsort(lower):
            if lower[index] > max(best, 0.):
                break
            value = mujoco.mj_geomDistance(self.model, self.data, self.pair_a[index], self.pair_b[index], 10., None)
            # 검사한 조각 쌍의 최소 거리를 누적: $$d_{new}=\min(d_{old},d_{ij})$$
            best = min(best, value)
            self.last_narrow_queries += 1
        return float(best)

    def label(self, q, *, progress=False):
        """관절 표본마다 같은 메시 기반 정답 함수를 호출한다."""
        values = []
        for index, sample in enumerate(q):
            values.append(self.distance(sample))
            if progress and (index + 1) % 1000 == 0:
                print(f"거리 라벨: {index + 1}/{len(q)}", flush=True)
        return np.asarray(values, dtype=np.float32)
