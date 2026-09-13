"""현재 XS 모델의 초기 자세를 MuJoCo에서 시각적으로 확인한다.
물리 시간을 진행하지 않고 관절각에 따른 자세만 갱신한다.
"""

import argparse
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parent / "models" / "xs" / "model.xml"


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """모델에 관절 표시용 좌표계를 붙이고 초기 관절각의 자세를 계산한다."""
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    for joint in spec.joints:
        # 관절점에 몸체의 로컬 좌표계를 붙이며 원본 XML에는 저장하지 않는다.
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
    mujoco.mj_forward(model, data)
    return model, data


def main() -> None:
    """초기 자세 뷰어를 열거나 창 없이 모델 로드만 확인한다."""
    parser = argparse.ArgumentParser(description="XS 로봇의 초기 자세 확인")
    parser.add_argument("--check", action="store_true", help="창 없이 모델 로드만 확인")
    args = parser.parse_args()

    model, data = load_model()
    print(f"모델 로드 완료: {MODEL_PATH}", flush=True)
    print(f"관절 {model.njnt}개 / 메시 {model.nmesh}개", flush=True)
    if args.check:
        return

    reset_requested = threading.Event()

    def on_key(keycode: int) -> None:
        """R 키를 누르면 다음 화면 갱신에서 초기 자세로 복귀하도록 요청한다."""
        if keycode in (ord("R"), ord("r")):
            reset_requested.set()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=True,
    ) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.cam.lookat[:] = data.body("forearm_R").xpos
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

        print(
            "RGB 축: 로컬 X/Y/Z / J1~J7: 팔 / G_L·G_R: 그리퍼\n"
            "tip_L·tip_R: 왼쪽·오른쪽 끝단 기준\n"
            "Joints: 관절각 조작 / R: 초기 자세 / 창 닫기: 종료",
            flush=True,
        )
        while viewer.is_running():
            with viewer.lock():
                if reset_requested.is_set():
                    reset_requested.clear()
                    mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
