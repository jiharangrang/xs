"""뎁스 녹화를 공통 편집표에 맞추고 녹화 종료 뒤에는 마지막 화면을 유지한다."""

from pathlib import Path
import subprocess
import tempfile

DEPTH_WIDTH = 284
DEPTH_HEIGHT = 174


def _run(command):
    """영상 도구가 실패하면 원래 오류를 전달한다."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)


def edit_depth_video(clip, edit, output):
    r"""로그 시각을 뎁스 재생 시각으로 바꾸어 같은 컷·배속으로 저장한다.

    $$t_{depth}=t_{log}-t_{depth,0}$$

    녹화 종료를 넘는 출력은 마지막 프레임으로 채운다.
    """
    output = Path(output).resolve()
    if min(cut.start for cut in edit.cuts) < clip.start:
        raise ValueError('뎁스 녹화 시작 이전의 편집 구간은 사용할 수 없습니다.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xs-depth-', dir=output.parent) as temporary:
        root = Path(temporary)
        final_frame = root / 'last.png'
        _run(['ffmpeg', '-v', 'error', '-y', '-sseof', '-1', '-i', str(clip.path),
              '-map', '0:v:0', '-an', '-vf', f'scale={DEPTH_WIDTH}:{DEPTH_HEIGHT},setsar=1',
              '-update', '1', str(final_frame)])
        parts = []
        for index, cut in enumerate(edit.cuts):
            print(f'뎁스 동기화 {index + 1}/{len(edit.cuts)}', flush=True)
            # 녹화 시작 기준의 뎁스 시각: $$t_{depth}=t_{log}-t_{depth,0}$$
            local_start = cut.start - clip.start
            command = ['ffmpeg', '-v', 'error', '-y']
            if local_start >= clip.duration:
                command.extend(['-loop', '1', '-framerate', str(edit.fps), '-i', str(final_frame)])
                filters = 'setsar=1'
            else:
                command.extend(['-ss', str(local_start), '-i', str(clip.path)])
                filters = (f'setpts=(PTS-STARTPTS)/{cut.speed},scale={DEPTH_WIDTH}:{DEPTH_HEIGHT},'
                           f'setsar=1,fps={edit.fps},tpad=stop_mode=clone:stop_duration={edit.duration + 1}')
            part = root / f'{index:03}.mp4'
            command.extend(['-map', '0:v:0', '-map_metadata', '-1', '-vf', filters, '-an',
                            '-frames:v', str(edit.frames[index]), '-c:v', 'libx264', '-threads', '2',
                            '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(part)])
            _run(command)
            parts.append(part)
        listing = root / 'parts.txt'
        listing.write_text(''.join(f"file '{part.name}'\n" for part in parts))
        _run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
              '-c', 'copy', '-movflags', '+faststart', str(output)])
    return output
