"""실영상과 시뮬레이션이 공유하는 컷·배속 편집표를 생성하고 병렬 영상을 저장한다."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from presentation.rendering import export_video, subtitle, _srt_time


@dataclass(frozen=True)
class Cut:
    """원본 로그 구간과 영상 번호, 표시 배속과 카메라를 지정한다."""

    clip: int
    start: float
    end: float
    speed: float
    camera: str = 'front'


class EditTimeline:
    """공백 없는 출력 시간에서 공통 로그 시각으로 역변환한다."""

    def __init__(self, cuts, *, fps=30.0, videos=()):
        r"""영상 프레임 수로 컷 경계를 확정해 누적 시각 오차를 방지한다.

        $$N_i=\lceil (e_i-s_i)f/v_i\rceil$$
        """
        self.cuts = list(cuts)
        self.fps = float(fps)
        self.videos = list(videos)
        if not self.cuts or not np.isfinite(fps) or fps <= 0:
            raise ValueError('유효한 컷과 프레임률이 필요합니다.')
        for cut in self.cuts:
            if not all(np.isfinite(v) for v in (cut.start, cut.end, cut.speed)) or cut.end <= cut.start or cut.speed <= 0:
                raise ValueError('편집 구간의 시각과 배속이 유효하지 않습니다.')
        # 각 컷의 출력 프레임 수: $$N_i=\lceil (e_i-s_i)f/v_i\rceil$$
        self.frames = [math.ceil((cut.end - cut.start) * fps / cut.speed) for cut in self.cuts]
        # 누적 프레임 수를 출력 초로 변환: $$b_i=\sum_{j<i}N_j/f$$
        self.boundaries = np.r_[0, np.cumsum(self.frames)] / fps
        self.duration = float(self.boundaries[-1])

    def sample(self, output_s):
        r"""편집 시각의 컷과 대응하는 원본 로그 시각을 반환한다.

        $$t_{log}=a_i+v_i(t_{edit}-b_i)$$
        """
        if not np.isfinite(output_s):
            raise ValueError('편집 시각은 유한해야 합니다.')
        index = int(np.clip(np.searchsorted(self.boundaries, output_s, side='right') - 1, 0, len(self.cuts) - 1))
        cut = self.cuts[index]
        # 해당 컷에서 경과한 로그 시간: $$\tau=v_i(t_{edit}-b_i)$$
        elapsed = cut.speed * max(0.0, output_s - self.boundaries[index])
        return min(cut.end, cut.start + elapsed), cut

    def save(self, path):
        """재편집·뷰어 재생에 재사용할 편집 데이터를 저장한다."""
        Path(path).write_text(json.dumps({'schema': 'xs.presentation.edit.v1', 'fps': self.fps, 'videos': self.videos,
                                         'cuts': [asdict(cut) for cut in self.cuts]}, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path):
        """저장한 편집표를 읽어 같은 시간 변환을 복원한다."""
        data = json.loads(Path(path).read_text())
        if data['schema'] != 'xs.presentation.edit.v1':
            raise ValueError('지원하지 않는 발표 편집표입니다.')
        return cls([Cut(**item) for item in data['cuts']], fps=data['fps'], videos=data['videos'])


def build_edit(choreography, clips, *, speed=5.0, fps=30.0):
    """실제 단계와 수동 파지를 남기고 긴 대기·촬영 공백을 제거한다."""
    rec = choreography.recording
    spans = [(rec.relative(rec.stage1['started_at_s']) - .5, rec.runs[0]['start_relative_walk_s'])]
    spans.extend((run['start_relative_walk_s'] - .3, run['end_relative_walk_s'] + .4) for run in rec.runs)
    for points in choreography.grips.values():
        for index in range(1, len(points), 2):
            begin, old = points[index]
            end, goal = points[index + 1]
            if goal == 0 and old < 0:
                spans.append((begin - .3, end + .4))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + .6:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    cuts = []
    for start, end in merged:
        for index, clip in enumerate(clips):
            first = max(start, clip.start, rec.start)
            last = min(end, clip.start + clip.duration, rec.end)
            if last <= first:
                continue
            # 마지막 파지를 완전히 담지 못하면 열린 상태의 도착까지만 사용한다.
            if first > rec.runs[-1]['end_relative_walk_s'] and last < end:
                continue
            cuts.append(Cut(index, first, last, speed, 'front'))
    cuts.sort(key=lambda cut: cut.start)
    return EditTimeline(cuts, fps=fps, videos=[{'path': str(clip.path), 'start_s': clip.start, 'duration_s': clip.duration} for clip in clips])


def _run(command):
    """영상 처리 실패를 숨기지 않고 도구의 오류 내용을 전달한다."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)


def assemble(choreography, edit, destination, *, overwrite=False):
    """같은 컷의 실영상·시뮬레이션 및 좌우 병렬 발표 영상을 만든다."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [destination / name for name in ('real.mp4', 'simulation.mp4', 'comparison.mp4', 'edit.json')]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('출력 폴더에 기존 영상이 있습니다. 새 폴더 또는 --overwrite를 사용해 주세요.')
    with tempfile.TemporaryDirectory(prefix='xs-edit-', dir=destination) as temporary:
        root = Path(temporary)
        real_parts, simulation_parts = [], []
        for index, cut in enumerate(edit.cuts):
            print(f'공통 편집 {index + 1}/{len(edit.cuts)}', flush=True)
            video = edit.videos[cut.clip]
            real = root / f'real_{index:03}.mp4'
            simulation = root / f'sim_{index:03}.mp4'
            local_start = cut.start - video['start_s']
            _run(['ffmpeg', '-v', 'error', '-y', '-ss', str(local_start), '-i', video['path'], '-map', '0:v:0', '-map_metadata', '-1', '-dn',
                  '-vf', f'setpts=(PTS-STARTPTS)/{cut.speed},scale=960:540,fps={edit.fps},tpad=stop_mode=clone:stop_duration=1',
                  '-frames:v', str(edit.frames[index]), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', str(real)])
            export_video(choreography, simulation, start=cut.start, end=cut.end, speed=cut.speed, fps=edit.fps,
                         width=960, height=540, camera_name=cut.camera, overlay=False)
            real_parts.append(real)
            simulation_parts.append(simulation)
        for parts, target in [(real_parts, outputs[0]), (simulation_parts, outputs[1])]:
            listing = root / (target.stem + '.txt')
            listing.write_text(''.join(f"file '{path.name}'\n" for path in parts))
            _run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(target)])
        compose_comparison(choreography, edit, destination)
        edit.save(outputs[3])
    return outputs


def compose_comparison(choreography, edit, destination):
    """완성된 두 영상에 공통 시각·단계 자막을 붙여 발표 화면을 합성한다."""
    destination = Path(destination).resolve()
    outputs = [destination / name for name in ('real.mp4', 'simulation.mp4', 'comparison.mp4')]
    captions = []
    for index, output_s in enumerate(np.arange(0, edit.duration, .2), 1):
        source_s, cut = edit.sample(output_s)
        stop = min(edit.duration, output_s + .2)
        captions.append(f'{index}\n{_srt_time(output_s)} --> {_srt_time(stop)}\n{subtitle(choreography, source_s, cut.speed)}\n')
    subtitle_path = destination / 'comparison.srt'
    subtitle_path.write_text('\n'.join(captions))
    font = '/System/Library/Fonts/Helvetica.ttc'
    graph = ("[0:v][1:v]hstack=inputs=2,pad=1920:1080:0:240:color=0xEAF0F5,"
             f"drawtext=fontfile={font}:text='XS INCHWORM':x=70:y=65:fontsize=52:fontcolor=0x263444,"
             f"drawtext=fontfile={font}:text='REAL FOOTAGE':x=70:y=177:fontsize=26:fontcolor=0x263444,"
             f"drawtext=fontfile={font}:text='SIMULATION RECONSTRUCTION':x=1030:y=177:fontsize=26:fontcolor=0x263444,"
             f"subtitles={subtitle_path}:force_style='Fontname=Helvetica,Fontsize=13,Alignment=2,MarginV=25,PrimaryColour=&H00443426,Outline=0'")
    _run(['ffmpeg', '-v', 'error', '-y', '-i', str(outputs[0]), '-i', str(outputs[1]), '-filter_complex', graph,
          '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(outputs[2])])
