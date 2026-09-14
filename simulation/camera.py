"""CAD 카메라 위치에서 지정 관절각의 RGB·깊이와 그리퍼 영역을 렌더링한다.
실제 스트림의 렌즈 설정을 적용하며 모터 연결이나 물리 시뮬레이션은 하지 않는다.
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import mujoco
import numpy as np
import yaml

from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
from kinematics.anchoring import TipAnchor
from simulation.model import apply_anchor


JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
REFERENCE_PATH = Path(__file__).resolve().parents[1] / "sensors/gemini215/camera_parameters.yaml"


@dataclass(frozen=True)
class SimCameraFrame:
    """서로 다른 광학 중심의 RGB·미터 깊이와 각 영상의 앞 그리퍼 영역이다."""

    rgb: np.ndarray
    depth_m: np.ndarray
    rgb_gripper_mask: np.ndarray
    depth_gripper_mask: np.ndarray
    camera_info: dict


def reference_camera_info() -> dict:
    """장치 없이 렌더링할 때 저장된 공장 렌즈 설정을 읽는다."""
    reference = yaml.safe_load(REFERENCE_PATH.read_text(encoding="utf-8"))
    profiles = {}
    for stream, name in (("color", "rgb"), ("depth", "depth")):
        profile = reference["device_profiles"][name]
        profiles[stream] = {
            "width": profile["resolution"][0], "height": profile["resolution"][1],
            "intrinsics": profile["intrinsics"], "distortion": profile["distortion"],
            "distortion_model": profile["distortion"]["model"],
        }
    return profiles


def _configure_projection(model: mujoco.MjModel, camera_id: int, profile: dict) -> None:
    r"""픽셀 렌즈 설정을 MuJoCo의 센서 중심 기준 투영 설정으로 옮긴다.

    $$I=(f_x/W,\ f_y/H,\ (W/2-c_x)/W,\ (H/2-c_y)/H)$$

    I는 cam_intrinsic, W와 H는 해상도, f와 c는 픽셀 초점거리와 주점이다.
    """
    width, height = profile["width"], profile["height"]
    lens = profile["intrinsics"]
    values = [width, height, *[lens[key] for key in ("fx", "fy", "cx", "cy")]]
    if not np.all(np.isfinite(values)) or min(width, height, lens["fx"], lens["fy"]) <= 0:
        raise ValueError("카메라 해상도와 초점거리는 유한한 양수여야 합니다.")
    model.cam_resolution[camera_id] = (width, height)
    model.cam_sensorsize[camera_id] = (1, 1)
    intrinsic = model.cam_intrinsic[camera_id]
    # 가로 초점거리를 센서 단위로 환산: $$I_0=f_x/W$$
    intrinsic[0] = lens["fx"] / width
    # 세로 초점거리를 센서 단위로 환산: $$I_1=f_y/H$$
    intrinsic[1] = lens["fy"] / height
    # MuJoCo의 가로 주점 부호를 적용: $$I_2=(W/2-c_x)/W$$
    intrinsic[2] = (width / 2 - lens["cx"]) / width
    # 렌더러의 세로 주점 부호를 적용: $$I_3=(H/2-c_y)/H$$
    intrinsic[3] = (height / 2 - lens["cy"]) / height


def _configure_color_pose(model: mujoco.MjModel, extrinsic: dict) -> None:
    r"""실측 깊이→RGB 보정값을 깊이 카메라의 CAD 장착 위치에 결합한다.

    $$p_{BC}=p_{BD}-R_{BD}R_{CD}^{T}t_{CD},\quad
    R_{BC}=R_{BD}R_{CD}^{T}$$

    B는 부모 몸체, D와 C는 깊이와 RGB 광학 좌표이며 t_CD의 입력 단위는 mm다.
    """
    depth, color = model.camera("gemini215_depth"), model.camera("gemini215_rgb")
    axis_flip = np.diag([1.0, -1.0, -1.0])
    rotation_gl = np.empty(9)
    mujoco.mju_quat2Mat(rotation_gl, depth.quat)
    # OpenGL 카메라축을 광학 좌표축으로 변환: $$R_{BD}=R_{B,GL}F$$
    rotation_bd = rotation_gl.reshape(3, 3) @ axis_flip
    rotation_cd = np.asarray(extrinsic["rotation"], dtype=float).reshape(3, 3)
    # 장치 내부 이동량을 미터로 환산: $$t_{CD}=t_{CD,mm}/1000$$
    translation_cd = np.asarray(extrinsic["translation_mm"], dtype=float).reshape(3) / 1000
    # RGB 광학 좌표의 부모 기준 회전: $$R_{BC}=R_{BD}R_{CD}^{T}$$
    rotation_bc = rotation_bd @ rotation_cd.T
    # 깊이 기준 위치에서 RGB 광학 중심을 계산: $$p_{BC}=p_{BD}-R_{BC}t_{CD}$$
    color.pos[:] = depth.pos - rotation_bc @ translation_cd
    # 광학 좌표를 렌더링 좌표로 변환: $$R_{B,GL_C}=R_{BC}F$$
    color_gl = rotation_bc @ axis_flip
    mujoco.mju_mat2Quat(color.quat, color_gl.reshape(-1))


def render_camera(
    joint_angles_deg: Mapping[str, float], *, camera_info: dict | None = None,
    include_beam: bool = False, anchor: TipAnchor | None = None,
) -> SimCameraFrame:
    r"""아홉 관절의 실측 각도를 적용해 RGB·깊이와 앞 그리퍼 마스크를 만든다.

    $$q_{rad}=q_{deg}\pi/180$$

    관측 각도를 그대로 사용하며 빔은 실제 배치를 맞춘 경우에만 표시한다.
    """
    if set(joint_angles_deg) != set(JOINT_NAMES):
        raise ValueError("J1~J7, G_L, G_R의 각도가 모두 필요합니다.")
    if not np.all(np.isfinite(list(joint_angles_deg.values()))):
        raise ValueError("관절각은 유한한 값이어야 합니다.")
    info = reference_camera_info() if camera_info is None else camera_info
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    for stream, name in (("color", "gemini215_rgb"), ("depth", "gemini215_depth")):
        _configure_projection(model, model.camera(name).id, info[stream])
    if "depth_to_color" in info:
        _configure_color_pose(model, info["depth_to_color"])
    model.vis.global_.offwidth = max(info[key]["width"] for key in ("color", "depth"))
    model.vis.global_.offheight = max(info[key]["height"] for key in ("color", "depth"))
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    # 광학 중심을 둘러싼 카메라 외장은 센서 시점에서 숨긴다.
    option.geomgroup[2] = 0
    if not include_beam:
        model.geom("ibeam_mesh").group[0] = 5
        option.geomgroup[5] = 0
    data = mujoco.MjData(model)
    for name in JOINT_NAMES:
        # 관절 제어 통로에서 받은 도 단위를 모델의 라디안으로 환산: $$q_{rad}=q_{deg}\pi/180$$
        data.joint(name).qpos[0] = np.deg2rad(joint_angles_deg[name])
    mujoco.mj_forward(model, data)
    if anchor is not None:
        apply_anchor(model, data, anchor)
    gripper_ids = [model.geom(name).id for name in ("gripper_R_geom_0", "gripper_R_thumb_geom_0")]
    images, masks = {}, {}
    for stream, name in (("color", "gemini215_rgb"), ("depth", "gemini215_depth")):
        profile = info[stream]
        with mujoco.Renderer(model, height=profile["height"], width=profile["width"]) as renderer:
            renderer.update_scene(data, camera=name, scene_option=option)
            if stream == "depth":
                renderer.enable_depth_rendering()
            images[stream] = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            is_geom = segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
            masks[stream] = is_geom & np.isin(segmentation[:, :, 0], gripper_ids)
            if stream == "depth":
                images[stream][~is_geom] = np.nan
    return SimCameraFrame(images["color"], images["depth"], masks["color"], masks["depth"], info)
