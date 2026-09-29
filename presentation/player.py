"""기록 시각으로 탐색·배속·카메라 전환이 가능한 독립 MuJoCo 재생 창을 제공한다."""

from queue import SimpleQueue
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from presentation.rendering import CAMERAS, set_camera, setup_scene, subtitle


def show(choreography, *, start, end, speed=1.0, camera_name='oblique', edit=None):
    """스페이스 재생, 방향키 탐색, 숫자 시점 전환으로 발표 영상을 미리 본다."""
    model, data, options = setup_scene(choreography)
    keys = SimpleQueue()
    if edit is not None:
        start, end = 0.0, edit.duration
    seconds = start
    playing = False
    speeds = [.25, .5, 1, 2, 4, 8, 16]
    marks = [run['start_relative_walk_s'] for run in choreography.recording.runs] if edit is None else list(edit.boundaries)
    previous_cut = None
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put, show_left_ui=False, show_right_ui=False) as viewer:
        set_camera(viewer.cam, camera_name)
        viewer.opt.sitegroup[:] = options.sitegroup
        last = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            if playing:
                seconds = min(end, seconds + (now - last) * speed)
                if seconds >= end:
                    playing = False
            last = now
            while not keys.empty():
                key = keys.get()
                if key == ord(' '):
                    if seconds >= end:
                        seconds = start
                    playing = not playing
                elif key in (ord('R'), ord('r')):
                    seconds, playing = start, False
                elif key == glfw.KEY_RIGHT:
                    seconds = min(end, seconds + 1)
                elif key == glfw.KEY_LEFT:
                    seconds = max(start, seconds - 1)
                elif key == glfw.KEY_UP:
                    speed = next((value for value in speeds if value > speed), speeds[-1])
                elif key == glfw.KEY_DOWN:
                    speed = next((value for value in reversed(speeds) if value < speed), speeds[0])
                elif key in (ord('N'), ord('n')):
                    seconds = next((mark for mark in marks if mark > seconds + .001), end)
                elif key in (ord('P'), ord('p')):
                    seconds = next((mark for mark in reversed(marks) if mark < seconds - .001), start)
                elif ord('1') <= key <= ord('5'):
                    set_camera(viewer.cam, list(CAMERAS)[key - ord('1')])
            source_seconds = seconds
            display_speed = speed
            if edit is not None:
                source_seconds, cut = edit.sample(seconds)
                display_speed *= cut.speed
                if previous_cut is None or cut.camera != previous_cut.camera:
                    set_camera(viewer.cam, cut.camera)
                previous_cut = cut
            with viewer.lock():
                viewer.opt.flags[:] = options.flags
                viewer.opt.geomgroup[:] = options.geomgroup
                viewer.opt.sitegroup[:] = options.sitegroup
                viewer.opt.frame = options.frame
                choreography.apply(model, data, source_seconds)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             'Space: play/pause | Left/Right: seek 1s\nUp/Down: speed | N/P: stage | R: reset\n1: front  2: oblique  3: rear  4: top  5: side',
                             subtitle(choreography, source_seconds, display_speed)))
            viewer.sync()
            time.sleep(.01)
