"""동일 편집표의 실영상과 세 시점을 사분면으로 배치한 발표 영상을 만든다.
시점은 고정하고 시간·단계 정보는 맨 아래의 얇은 띠에 표시한다.
"""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile

import mujoco
import numpy as np

from presentation.rendering import CAMERAS, STAGES, set_camera, setup_scene

PANEL_WIDTH = 960
PANEL_HEIGHT = 516
FOOTER_HEIGHT = 48
TOP_BEAM_ALPHA = .20
VIEW_LABEL_SIZE = 26


def validate_video(path, edit):
    """합성 입력이 공통 편집표와 같은 프레임 수·프레임률인지 확인한다."""
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=nb_frames,r_frame_rate', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    numerator, denominator = map(float, stream['r_frame_rate'].split('/'))
    if int(stream['nb_frames']) != sum(edit.frames) or not np.isclose(numerator / denominator, edit.fps):
        raise ValueError(f'공통 편집표와 영상의 프레임 수 또는 속도가 다릅니다: {path}')


def render_edit_views(choreography, edit, directory, *, views=('side', 'top')):
    r"""공통 출력 프레임에서 원본 시각을 한 번 계산해 추가 시점들을 렌더링한다.

    $$t_k=E(k/f)$$

    E는 저장된 컷·배속을 반영하는 편집 시각의 역변환이다.
    SIDE는 현재 FRONT를 사용자 기준 방위각 180도로 둔 상대 방위각 50도이며,
    MuJoCo 방위각은 -40도, 고도는 -20도다. TOP의 월드 양의 X축은 화면 오른쪽이다.
    """
    directory = Path(directory)
    model, data, options = setup_scene(choreography, PANEL_WIDTH, 540)
    cameras = {}
    for name in views:
        camera = mujoco.MjvCamera()
        set_camera(camera, name)
        cameras[name] = camera
    beam = model.geom('ibeam_mesh')
    frame_count = sum(edit.frames)
    outputs = {name: directory / f'{name}.mp4' for name in views}
    processes = {}
    with ExitStack() as stack:
        errors = {name: stack.enter_context(tempfile.TemporaryFile()) for name in views}
        try:
            for name, output in outputs.items():
                command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x540',
                           '-r', str(edit.fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4',
                           '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
                processes[name] = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors[name])
            with mujoco.Renderer(model, height=540, width=PANEL_WIDTH) as renderer:
                for index in range(frame_count):
                    # 출력 프레임 번호의 편집 시각: $$t_{edit}=k/f$$
                    output_s = index / edit.fps
                    source_s, _ = edit.sample(output_s)
                    choreography.apply(model, data, source_s)
                    for name, camera in cameras.items():
                        beam.rgba[3] = TOP_BEAM_ALPHA if name == 'top' else 1.0
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        processes[name].stdin.write(renderer.render().tobytes())
                    if index % max(1, round(edit.fps * 5)) == 0:
                        print(f'추가 시점 저장 {index / frame_count:.0%}', flush=True)
            for process in processes.values():
                process.stdin.close()
            for name, process in processes.items():
                if process.wait() != 0:
                    errors[name].seek(0)
                    raise RuntimeError(errors[name].read().decode())
        except BaseException:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
                process.wait()
            for error in errors.values():
                error.seek(0)
                detail = error.read().decode()
                if detail:
                    print(detail)
            raise
    return outputs


def _ass_time(seconds):
    """자막 시각을 ASS의 백분의 일 초 표기로 바꾼다."""
    total = round(seconds * 100)
    hours, rest = divmod(total, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, centis = divmod(rest, 100)
    return f'{hours}:{minutes:02}:{secs:02}.{centis:02}'


def footer_captions(choreography, edit, path):
    r"""시작 기준 경과 시간과 배속, 빨간 굵은 상태를 하단 한 줄로 저장한다.

    $$t_{relative}=t_{source}-t_{first}$$

    잘라낸 대기 구간은 건너뛰고 원본 기록의 경과 시간을 표시한다.
    """
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Footer,Helvetica,26,&H0028211C,&H0028211C,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    lines = []
    for output_s in np.arange(0, edit.duration, .2):
        source_s, cut = edit.sample(output_s)
        # 첫 영상 프레임을 기준으로 한 원본 기록 경과 시간: $$t_{relative}=t_{source}-t_{first}$$
        relative_s = source_s - edit.cuts[0].start
        milliseconds = round(relative_s * 1000)
        minutes, remainder = divmod(milliseconds, 60000)
        seconds, fraction = divmod(remainder, 1000)
        elapsed = f'{minutes:02}:{seconds:02}.{fraction:03}'
        stage, state = choreography.recording.status(source_s)
        status = f'{STAGES[stage]} / {state}'
        status = status.replace('{', '').replace('}', '').replace('\\', '')
        line = f'{elapsed}  |  {cut.speed:g}x  |  {{\\c&H2828D6&\\b1}}{status}{{\\r}}'
        end = min(edit.duration, output_s + .2)
        lines.append(f'Dialogue: 0,{_ass_time(output_s)},{_ass_time(end)},Footer,,0,0,0,,{{\\pos(960,1056)}}{line}\n')
    Path(path).write_text(header + ''.join(lines))


def four_view_filter(*, depth=False):
    """네 화면의 이름과 선택한 뎁스 창을 배치하고 하단 정보 띠를 붙인다."""
    font = '/System/Library/Fonts/Helvetica.ttc'
    filters = []
    for index, label in enumerate(('REAL', 'FRONT', 'SIDE', 'TOP')):
        filters.append(
            f'[{index}:v]scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
            f'pad={PANEL_WIDTH}:{PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0xEAF0F5,setsar=1,'
            f"drawtext=fontfile={font}:text='{label}':fontcolor=black:fontsize={VIEW_LABEL_SIZE}:x=(w-tw)/2:y=h-th-10:"
            f'box=1:boxcolor=0xEAF0F5@0.72:boxborderw=4[v{index}]')
    side = 'v2'
    if depth:
        # 원본 화면의 작은 DEPTH 배지를 지우고 다른 시점과 같은 크기로 아래에 표시한다.
        filters.append('[4:v]drawbox=x=4:y=2:w=40:h=22:color=black:t=fill,'
                       'drawbox=x=0:y=0:w=iw:h=ih:color=0xB8C5CE:t=2,'
                       f"drawtext=fontfile={font}:text='DEPTH':fontcolor=white:fontsize={VIEW_LABEL_SIZE}:"
                       'x=10:y=h-th-10[depth]')
        filters.append('[v2][depth]overlay=x=14:y=main_h-overlay_h-14:shortest=1[v2depth]')
        side = 'v2depth'
    filters.append(f'[v0][v1][{side}][v3]xstack=inputs=4:layout=0_0|960_0|0_516|960_516,'
                   'pad=1920:1080:0:0:color=0xEAF0F5,ass=footer.ass')
    return ';'.join(filters)


def compose_four_view(choreography, edit, source_directory, output, *, overwrite=False, depth_clip=None):
    """같은 시간의 REAL·FRONT·SIDE·TOP과 얇은 정보 띠를 한 영상으로 합친다."""
    source_directory = Path(source_directory).resolve()
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f'덮어쓰려면 --overwrite를 지정해 주세요: {output}')
    if output in {source_directory / 'real.mp4', source_directory / 'simulation.mp4'}:
        raise ValueError('출력 파일은 실영상·정면 입력과 다른 이름이어야 합니다.')
    real, front = source_directory / 'real.mp4', source_directory / 'simulation.mp4'
    for path in (real, front):
        validate_video(path, edit)
    if any(cut.camera != 'front' for cut in edit.cuts):
        raise ValueError('FRONT 입력에는 전체 정면 시점으로 고정한 편집표가 필요합니다.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xs-quad-', dir=output.parent) as temporary:
        root = Path(temporary)
        views = render_edit_views(choreography, edit, root)
        for path in views.values():
            validate_video(path, edit)
        depth = None
        if depth_clip is not None:
            from presentation.depth_overlay import edit_depth_video
            depth = edit_depth_video(depth_clip, edit, root / 'depth.mp4')
            validate_video(depth, edit)
        footer_captions(choreography, edit, root / 'footer.ass')
        command = ['ffmpeg', '-v', 'error', '-y']
        inputs = [real, front, views['side'], views['top']]
        if depth is not None:
            inputs.append(depth)
        for path in inputs:
            command.extend(['-i', str(path)])
        command.extend(['-filter_complex_threads', '2', '-filter_complex', four_view_filter(depth=depth is not None), '-an',
                        '-frames:v', str(sum(edit.frames)), '-r', str(edit.fps), '-fps_mode', 'cfr',
                        '-c:v', 'libx264', '-threads', '4', '-preset', 'fast',
                        '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', 'four_view.mp4'])
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        validate_video(root / 'four_view.mp4', edit)
        (root / 'four_view.mp4').replace(output)
        (root / 'footer.ass').replace(output.with_suffix('.ass'))
        for name, path in views.items():
            path.replace(output.with_name(f'{output.stem}_{name}.mp4'))
        if depth is not None:
            depth.replace(output.with_name(f'{output.stem}_depth.mp4'))
            sync = {'depth_path': str(depth_clip.path), 'depth_start_log_s': depth_clip.start,
                    'depth_duration_s': depth_clip.duration, 'after_depth_end': 'hold_last_frame',
                    'side_camera': CAMERAS['side']}
            output.with_suffix('.sync.json').write_text(json.dumps(sync, ensure_ascii=False, indent=2))
    return output
