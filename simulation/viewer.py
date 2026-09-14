"""관절각 자세 비교와 고정단을 반영한 경로 재생을 같은 MuJoCo 뷰어에서 제공한다."""

from collections.abc import Sequence
from pathlib import Path
import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import ArrayLike

from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.joints import as_joint_angles
from planning.motion_path import MotionPath
from simulation.model import load_model, set_arm_angles, set_path_time


def show_pose(
    q_rad: ArrayLike, model_path: str | Path = DEFAULT_MODEL_PATH,
    *, other_candidates: Sequence[ArrayLike] | None = None,
) -> None:
    """입력 관절각의 자세를 표시하고 창을 닫을 때까지 관절 조작을 지원한다.

    q_rad는 J1부터 J7 순서의 라디안 배열이다.
    macOS에서는 호출하는 스크립트를 mjpython으로 실행해야 한다.
    R 키는 입력 자세로 돌아가고, 스페이스는 입력 자세와 XML 초기 자세를 전환한다.
    other_candidates를 전달하면 q_rad를 후보 1로 두고 나머지를 Enter 키로 순환한다.
    후보 모드의 R 키는 선택한 후보를 복원하며 스페이스도 그 후보와 초기 자세를 비교한다.
    """
    _show_viewer(q_rad, model_path, other_candidates=other_candidates)


def show_path(motion_path: MotionPath, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
    """저장된 경로의 첫 자세를 표시하고 Space로 재생·일시정지, R로 처음으로 돌아간다.

    마지막 자세에서는 자동으로 멈춘다. 끝에서 Space를 누르면 처음부터 재생한다.
    macOS에서는 mjpython으로 실행해야 한다.
    """
    _show_viewer(motion_path.segments[0].q_rad[0], model_path, motion_path=motion_path)


def _show_viewer(
    q_rad: ArrayLike, model_path: str | Path,
    *, other_candidates: Sequence[ArrayLike] | None = None, motion_path: MotionPath | None = None,
) -> None:
    """공통 표시 설정과 창을 사용하며 자세 비교 또는 경로 재생 모드로 화면을 갱신한다."""
    input_angles = as_joint_angles(q_rad)
    candidate_mode = other_candidates is not None
    poses = [input_angles]
    if candidate_mode:
        poses.extend(as_joint_angles(angles) for angles in other_candidates)
    candidate_index = 0
    model, data = load_model(input_angles, model_path)
    restore_requested = threading.Event()
    toggle_requested = threading.Event()
    next_requested = threading.Event()
    showing_input = True
    elapsed_s = 0.0
    playing = False
    segment_index = 0
    if motion_path is not None:
        segment_index = set_path_time(model, data, motion_path, elapsed_s)

    def on_key(keycode: int) -> None:
        """다음 화면 갱신에서 입력 자세 또는 초기 자세로 돌아가도록 요청한다."""
        if keycode in (ord("R"), ord("r")):
            restore_requested.set()
        elif keycode == ord(" "):
            toggle_requested.set()
        elif candidate_mode and keycode == glfw.KEY_ENTER:
            next_requested.set()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key, show_left_ui=False, show_right_ui=True
    ) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.cam.lookat[:] = data.body("forearm_R").xpos
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
            # 카메라 보조 사이트의 삼축 표시는 숨기고 렌즈 방향선으로 대체한다.
            viewer.opt.sitegroup[3] = 0
        def update_texts() -> None:
            """현재 모드의 키 안내와 선택 자세 또는 재생 시각을 갱신한다."""
            if motion_path is not None:
                segment = motion_path.segments[segment_index]
                controls = "Space: play / pause | R: first frame"
                state = "Playing" if playing else "Paused"
                status = f"{state} {elapsed_s:.2f} / {motion_path.duration_s:.2f} s"
                status += f"\n{segment.name} | fixed: {segment.anchor.fixed_tip}"
            else:
                controls = "Space: input / initial | R: input pose\nJoint: edit angles"
                status = "Input pose" if showing_input else "XML initial pose"
            if model.ncam:
                controls += "\nLens: blue = Depth | orange = RGB"
            if candidate_mode:
                controls = "Enter: next candidate\n" + controls
                status = f"Candidate {candidate_index + 1}/{len(poses)} | {status}"
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                controls,
                status + "\nKinematics only",
            ))

        update_texts()
        # 표시할 각도를 도 단위로 변환: $$q_{\mathrm{deg}}=q_{\mathrm{rad}}180/\pi$$
        input_degrees = np.rad2deg(input_angles)
        print("입력 관절각(deg):", input_degrees, flush=True)
        if motion_path is not None:
            print("Space: 재생·일시정지 / R: 처음으로 (일시정지)", flush=True)
        else:
            print("Space: 입력·초기 자세 전환 / R: 입력 자세 / Joint: 관절각 조작", flush=True)
        if candidate_mode:
            print(f"후보 1/{len(poses)} / Enter: 다음 후보 (마지막 다음은 후보 1)", flush=True)
        last_update = time.monotonic()
        while viewer.is_running():
            display_changed = False
            with viewer.lock():
                now = time.monotonic()
                if motion_path is not None:
                    # 재생 중 경과 시간 누적: $$t_{\mathrm{new}}=t_{\mathrm{old}}+\Delta t$$
                    elapsed_s += now - last_update if playing else 0.0
                    if elapsed_s >= motion_path.duration_s:
                        elapsed_s = motion_path.duration_s
                        playing = False
                    if toggle_requested.is_set():
                        toggle_requested.clear()
                        if elapsed_s >= motion_path.duration_s:
                            elapsed_s = 0.0
                        playing = not playing
                    if restore_requested.is_set():
                        restore_requested.clear()
                        elapsed_s = 0.0
                        playing = False
                    segment_index = set_path_time(model, data, motion_path, elapsed_s)
                    display_changed = True
                last_update = now
                if next_requested.is_set():
                    next_requested.clear()
                    candidate_index = (candidate_index + 1) % len(poses)
                    input_angles = poses[candidate_index]
                    restore_requested.set()
                    # 다음 후보의 각도를 도 단위로 변환: $$q_{\mathrm{deg}}=q_{\mathrm{rad}}180/\pi$$
                    input_degrees = np.rad2deg(input_angles)
                    print(f"후보 {candidate_index + 1}/{len(poses)} 관절각(deg): {input_degrees}", flush=True)
                if toggle_requested.is_set():
                    toggle_requested.clear()
                    display_changed = True
                    showing_input = not showing_input
                    mujoco.mj_resetData(model, data)
                    if showing_input:
                        set_arm_angles(model, data, input_angles)
                if restore_requested.is_set():
                    restore_requested.clear()
                    display_changed = True
                    showing_input = True
                    mujoco.mj_resetData(model, data)
                    set_arm_angles(model, data, input_angles)
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                _add_camera_rays(model, data, viewer.user_scn)
            if display_changed:
                update_texts()
            viewer.sync()
            time.sleep(0.02)


def _add_camera_rays(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: mujoco.MjvScene,
    *,
    length_m: float = 0.05,
) -> None:
    r"""Depth와 RGB 광학 중심에서 촬영 방향으로 짧은 선을 추가한다.

    $$
    \mathbf{p}_{\mathrm{end}} = \mathbf{p}_C - \ell R_C[:,2]
    $$

    \(\mathbf{p}_C\)와 \(R_C\)는 월드에서 본 MuJoCo 카메라의 위치와 회전이고,
    \(\ell\)은 length_m으로 지정하는 선 길이이다. MuJoCo 카메라는 로컬 음의 Z축을 본다.
    data는 mj_forward로 갱신된 상태여야 하며 기존 scene 도형은 유지한다.
    해당 이름의 카메라가 없는 모델에서는 보조선을 추가하지 않는다.
    """
    camera_colors = (
        ("gemini215_depth", [0.1, 0.7, 1.0, 1.0]),
        ("gemini215_rgb", [1.0, 0.4, 0.1, 1.0]),
    )
    for name, color in camera_colors:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0 or scene.ngeom >= scene.maxgeom:
            continue
        position = data.cam_xpos[camera_id]
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        # 렌즈 앞쪽으로 표시할 끝점: $$\mathbf{p}_{\mathrm{end}}=\mathbf{p}_C-\ell R_C[:,2]$$
        endpoint = position - length_m * rotation[:, 2]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, position, endpoint,
        )
        scene.ngeom += 1
