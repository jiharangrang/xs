"""같은 6단계 경로를 두 화면에 재생하고 메시·학습 거리와 실제 조회 시간을 표시한다.
재생 간격은 두 화면이 같으며 소규모 모델의 정확도 미검증 표시를 유지한다.
"""

import argparse
import os
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from common import ARTIFACTS
from model import load_checkpoint
from scene import DistanceScene


def render_frame(scene, renderer, q, view="overview"):
    """하드웨어 접속 없이 현재 관절각의 MuJoCo 영상을 만든다."""
    scene.set_q(q)
    mujoco.mj_forward(scene.model, scene.data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (scene.data.site("tip_L").xpos + scene.data.site("tip_R").xpos) / 2
    camera.distance = .95
    camera.azimuth = 140
    camera.elevation = -10
    if view == "fixed":
        camera.lookat[:] = scene.data.site("tip_L").xpos
        camera.distance = .28
        camera.elevation = -15
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    renderer.update_scene(scene.data, camera=camera, scene_option=option)
    return renderer.render().copy()


def configure_rendering(scene):
    """검은 부품이 배경에 묻히지 않도록 데모의 조명과 표시 색만 밝힌다."""
    scene.model.vis.headlight.ambient[:] = .65
    scene.model.vis.headlight.diffuse[:] = .9
    dark = np.all(scene.model.geom_rgba[:, :3] < .1, axis=1)
    scene.model.geom_rgba[dark, :3] = [.25, .28, .33]


def main():
    """동기화한 두 화면을 재생하거나 지정 표본의 비교 화면을 저장한다."""
    parser = argparse.ArgumentParser(description="메시와 학습 충돌거리 비교 시뮬레이션")
    parser.add_argument("--scene", type=Path, default=ARTIFACTS / "scene")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "smoke/active.pt")
    parser.add_argument("--snapshot", type=Path, help="창 대신 비교 PNG 한 장 저장")
    parser.add_argument("--sample", type=int, default=0, help="스냅샷 표본; 기본값은 초기 파지 자세")
    parser.add_argument("--view", choices=("overview", "fixed"), default="overview", help="전체 자세 또는 왼쪽 고정 파지 확대")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / ".mpl-cache"))
    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    scene = DistanceScene(args.scene)
    configure_rendering(scene)
    model, checkpoint = load_checkpoint(args.checkpoint, scene)
    with np.load(args.scene / "trace.npz", allow_pickle=False) as trace:
        q, times, stages = trace["q"], trace["time_s"], trace["stage"]
    if args.snapshot and not -len(q) <= args.sample < len(q):
        parser.error("표본 번호가 경로 범위를 벗어났습니다.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    artists = [axis.imshow(np.zeros((480, 640, 3), dtype=np.uint8)) for axis in axes]
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Ideal CAD path / left gripper fixed / model NOT accuracy-qualified")
    try:
        with mujoco.Renderer(scene.model, height=480, width=640) as renderer, torch.inference_mode():
            for sample in q[:8]:
                scene.distance(sample)
                model(torch.tensor(sample[None], dtype=torch.float32))
            indices = [args.sample % len(q)] if args.snapshot else range(len(q))
            for index in indices:
                start = time.perf_counter()
                true_distance = scene.distance(q[index])
                mesh_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                learned = float(model(torch.tensor(q[index:index + 1], dtype=torch.float32))[0])
                learned_ms = (time.perf_counter() - start) * 1000
                rgb = render_frame(scene, renderer, q[index], args.view)
                for artist in artists:
                    artist.set_data(rgb)
                axes[0].set_title(f"Mesh | stage {stages[index]} | t={times[index]:.1f} s\n{true_distance * 1000:.2f} mm | {mesh_ms:.2f} ms/query")
                axes[1].set_title(f"Learned ({checkpoint['config']['profile']})\n{learned * 1000:.2f} mm | {learned_ms:.2f} ms/query")
                if args.snapshot:
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(args.snapshot, dpi=130)
                else:
                    if not plt.fignum_exists(figure.number):
                        break
                    plt.pause(.1)
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
