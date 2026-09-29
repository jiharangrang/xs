"""발표용 모델의 카메라와 화면을 구성하고 동일 시간축의 MP4를 저장한다."""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np


CAMERAS = {'front': (90, -4, .52), 'oblique': (65, -12, .56), 'rear': (-90, -10, .55),
           'top': (90, -90, .62), 'side': (-40, -20, .60)}
STAGES = {'READY': 'Ready', 'STAGE 1': 'Observation pose', 'STAGE 2': 'Sag / tilt correction', 'STAGE 3': 'Approach height', 'STAGE 4': 'Side clearance', 'STAGE 5': 'Final lift', 'STAGE 6': 'Insert gripper', 'STAGE 7': 'Rear pull', 'STAGE 8': 'Front advance'}


def setup_scene(choreography, width=1280, height=720):
    """녹화 당시 모델을 표시용으로 로드하고 보조 좌표축을 숨긴다."""
    spec = mujoco.MjSpec.from_file(str(choreography.recording.model_path))
    spec.add_texture(name='presentation_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.93, .95, .97], rgb2=[.82, .87, .92], width=64, height=384)
    spec.add_material(name='presentation_charcoal', rgba=[.23, .27, .32, 1], specular=.45, shininess=.35)
    for geom in spec.geoms:
        if max(geom.rgba[:3]) < .15:
            geom.material = 'presentation_charcoal'
            geom.rgba = [.23, .27, .32, 1]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.headlight.ambient[:] = [.55, .55, .55]
    model.vis.headlight.diffuse[:] = [.7, .7, .7]
    model.vis.headlight.specular[:] = [.2, .2, .2]
    model.geom('ibeam_mesh').rgba[:] = [.33, .37, .43, 1]
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.frame = mujoco.mjtFrame.mjFRAME_NONE
    return model, data, options


def set_camera(camera, name):
    """발표 중 같은 구도를 유지하는 카메라 프리셋을 적용한다."""
    azimuth, elevation, distance = CAMERAS[name]
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.16, -.035, .105]
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance


def subtitle(choreography, seconds, speed=1.0):
    """한국 시각과 원본 단계 상태를 표시용 영문 문자열로 만든다."""
    stage, state = choreography.recording.status(seconds)
    wall = choreography.recording.wall_time(seconds).strftime('%H:%M:%S.%f')[:-3]
    return f'{wall} KST   |   {speed:g}x\n{STAGES[stage]}  /  {state}\nPresentation reconstruction'


def _srt_time(seconds):
    """초 단위 시간을 자막 형식으로 변환한다."""
    total = round(seconds * 1000)
    hours, rest = divmod(total, 3600000)
    minutes, rest = divmod(rest, 60000)
    secs, millis = divmod(rest, 1000)
    return f'{hours:02}:{minutes:02}:{secs:02},{millis:03}'


def export_video(choreography, output, *, start, end, speed=1.0, fps=30.0, width=1280, height=720, camera_name='oblique', overlay=True, overwrite=False):
    r"""프레임 번호에서 실제 로그 시각을 계산해 배속에 독립적인 영상을 저장한다.

    $$t_k=t_0+k v/f$$
    """
    if not all(np.isfinite(v) for v in (start, end, speed, fps)) or end <= start or speed <= 0 or fps <= 0:
        raise ValueError('내보내기 시각·배속·프레임률을 확인해 주세요.')
    if width < 64 or height < 64 or width % 2 or height % 2:
        raise ValueError('영상 가로·세로는 64 이상의 짝수여야 합니다.')
    if not shutil.which('ffmpeg'):
        raise ValueError('MP4 저장에는 ffmpeg가 필요합니다.')
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f'기존 파일을 덮어쓰려면 --overwrite를 지정해 주세요: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    model, data, options = setup_scene(choreography, width, height)
    camera = mujoco.MjvCamera()
    set_camera(camera, camera_name)
    # 출력 프레임 수: $$N=\lceil(t_1-t_0)f/v\rceil$$
    frame_count = math.ceil((end - start) * fps / speed)
    with tempfile.TemporaryDirectory(prefix='xs-video-', dir=output.parent) as directory:
        destination = Path(directory) / 'render.mp4'
        command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an']
        if overlay:
            command.extend(['-vf', "drawtext=fontfile=/System/Library/Fonts/Helvetica.ttc:text='Presentation reconstruction':fontcolor=white:fontsize=20:x=28:y=28:box=1:boxcolor=black@0.5:boxborderw=10"])
        command.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)])
        subtitles = []
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                with mujoco.Renderer(model, height=height, width=width) as renderer:
                    for index in range(frame_count):
                        # 배속을 적용한 원본 로그 시각: $$t_k=t_0+kv/f$$
                        seconds = start + index * speed / fps
                        choreography.apply(model, data, seconds)
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        process.stdin.write(renderer.render().tobytes())
                        if index % max(1, round(fps)) == 0:
                            print(f'영상 저장 {index / frame_count:.0%}: {output.name}', flush=True)
                        if index % max(1, round(fps / 5)) == 0:
                            begin = index / fps
                            stop = min(frame_count / fps, (index + max(1, round(fps / 5))) / fps)
                            subtitles.append(f'{len(subtitles)+1}\n{_srt_time(begin)} --> {_srt_time(stop)}\n{subtitle(choreography, seconds, speed)}\n')
                process.stdin.close()
                if process.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode())
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                errors.seek(0)
                detail = errors.read().decode()
                if detail:
                    print(detail)
                raise
        destination.replace(output)
        output.with_suffix('.srt').write_text('\n'.join(subtitles))
    return output
