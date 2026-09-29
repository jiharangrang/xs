"""원본 촬영 시각과 발표 시간축을 연결하고 클립별 내보내기 범위를 정한다."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class VideoClip:
    """원본 클립의 시간 범위와 프레임률을 보관한다."""

    path: Path
    start: float
    duration: float
    fps: float

    def local_time(self, seconds):
        r"""로그 상대 시각을 원본 클립의 재생 초로 바꾼다.

        $$t_v=t-t_{v,0}$$
        """
        # 클립 시작을 뺀 원본 영상 시각: $$t_v=t-t_{v,0}$$
        return seconds - self.start


def probe_video(path, recording, offset_s=0.0):
    """촬영 메타데이터와 사용자가 지정한 시계 오프셋으로 클립 위치를 계산한다."""
    path = Path(path).expanduser().resolve()
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    video = next(stream for stream in metadata['streams'] if stream['codec_type'] == 'video')
    created = metadata['format'].get('tags', {}).get('creation_time') or video.get('tags', {}).get('creation_time')
    if not created:
        raise ValueError(f'촬영 시각 메타데이터가 없습니다: {path.name}')
    wall_start = datetime.fromisoformat(created.replace('Z', '+00:00')) + timedelta(seconds=offset_s)
    start = (wall_start - recording.wall_origin).total_seconds()
    return VideoClip(path, start, float(video.get('duration', metadata['format']['duration'])), float(Fraction(video['r_frame_rate'])))
