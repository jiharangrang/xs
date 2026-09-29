"""독립적인 ㄱ자 빔 경로 생성·검사·영상 저장·대화형 재생을 실행한다.
프로젝트 루트에서 uv run --project Corner python -m Corner.demo 명령으로 사용한다.
"""

import argparse
import time

from Corner.planner import Planner, Settings
from Corner.scene import OUTPUT, Scene, build_scene
from Corner.trajectory import Trajectory
from Corner.validate import validate


def view():
    """저장한 경로를 MuJoCo 창에서 반복 재생한다."""
    import mujoco.viewer

    trajectory = Trajectory()
    scene = Scene(OUTPUT / "scene/model.xml")
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = [.10, .12, .18]
        viewer.cam.distance = .85
        viewer.cam.azimuth = 132
        viewer.cam.elevation = -18
        started = time.monotonic()
        while viewer.is_running():
            elapsed = (time.monotonic() - started) % trajectory.duration
            q, grippers, anchor, _ = trajectory.sample(elapsed)
            with viewer.lock():
                scene.set(q, grippers, anchor)
            viewer.sync()
            time.sleep(1 / 60)


def main():
    """요청한 시뮬레이션 작업을 수행하며 장치 통신 코드는 불러오지 않는다."""
    parser = argparse.ArgumentParser(description="ㄱ자 빔 전체 전환의 시뮬레이션 전용 데모")
    parser.add_argument("action", choices=("all", "plan", "validate", "render", "view"), nargs="?", default="all")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.action in ("all", "plan"):
        settings = Settings()
        scene = Scene(build_scene(settings.corner_x))
        planner = Planner(scene, settings)
        planner.plan()
        planner.save()
    if args.action in ("all", "plan", "validate"):
        validate()
    if args.action in ("all", "render"):
        from Corner.render import render
        render(fps=args.fps)
    if args.action == "view":
        view()


if __name__ == "__main__":
    main()
