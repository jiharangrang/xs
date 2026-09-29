"""녹화 묶음에서 단계 시각과 당시 모델만 읽는다.
저장된 제어 코드를 실행하거나 실물 장치에 연결하지 않는다.
"""

from contextlib import contextmanager
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

DEFAULT_RECORDING = Path(__file__).resolve().parents[1] / 'outputs/walk_recordings/20260919_144803_kst_two_cycles.tar.gz'


@contextmanager
def recording_directory(source):
    """폴더 또는 압축 묶음에서 필요한 데이터만 임시 폴더로 준비한다."""
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        yield source
        return
    with tempfile.TemporaryDirectory(prefix='xs-presentation-') as temporary:
        root = Path(temporary)
        with tarfile.open(source, 'r:gz') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or not member.isfile():
                    continue
                relative = PurePosixPath(*parts[1:])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('녹화 파일에 허용되지 않는 경로가 있습니다.')
                name = str(relative)
                selected = name in {'manifest.json', 'timeline.json', 'sync_events.csv', 'status_at_archive.json'}
                selected |= name.startswith('snapshot/models/xs/')
                selected |= name == 'snapshot/outputs/stage1_observation_pose.json'
                selected |= name.startswith('raw/outputs/stage') and name.endswith('.jsonl')
                if selected:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as stream:
                        destination.write_bytes(stream.read())
        yield root


class Recording:
    """단조 시각을 공통 상대 시각으로 바꾸고 원본 실행 결과를 보관한다."""

    def __init__(self, root):
        """필수 기록과 모델의 해시를 확인하고 이벤트를 읽는다."""
        self.root = Path(root)
        self.manifest = self.read_json('manifest.json')
        self.timeline = self.read_json('timeline.json')
        if self.manifest['schema'] != 'xs.walk_recording.v1':
            raise ValueError('지원하지 않는 녹화 형식입니다.')
        for item in self.manifest['copied_files']:
            path = self.root / item['saved_as']
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'녹화 스냅샷 해시 불일치: {item["saved_as"]}')
        self.origin = float(self.timeline['time_origin_monotonic_s'])
        self.wall_origin = datetime.fromisoformat(self.manifest['clock']['relative_walk_zero_time_kst'])
        self.runs = sorted(self.timeline['runs'], key=lambda run: run['start_monotonic_s'])
        self.rows = {run['run_id']: [json.loads(line) for line in (self.root / run['log_file']).read_text().splitlines()] for run in self.runs}
        with (self.root / 'sync_events.csv').open() as stream:
            self.sync = list(csv.DictReader(stream))
        self.stage1 = self.read_json('status_at_archive.json')['stages']['1']
        self.observation_pose = self.read_json('snapshot/outputs/stage1_observation_pose.json')
        self.model_path = self.root / 'snapshot/models/xs/model.xml'
        self.calibration_path = self.root / 'snapshot/models/xs/calibration.yaml'
        self.start = self.relative(self.manifest['clock']['origin_monotonic_s'])
        self.end = self.relative(self.manifest['clips']['extended_pose_through_walk']['end_monotonic_s'])

    def read_json(self, name):
        """묶음 안의 JSON 데이터를 읽는다."""
        return json.loads((self.root / name).read_text())

    def relative(self, monotonic_s):
        r"""기록 시각을 6단계 완료 기준의 상대 초로 바꾼다.

        $$t=m-m_0$$
        """
        # 단조 시각의 기준 이동: $$t=m-m_0$$
        return float(monotonic_s) - self.origin

    def wall_time(self, seconds):
        """상대 시각에 해당하는 한국 시각을 반환한다."""
        return self.wall_origin + timedelta(seconds=float(seconds))

    def status(self, seconds):
        """해당 시각의 실제 단계와 원본 결과를 반환한다."""
        if seconds < self.relative(self.stage1['started_at_s']):
            return 'READY', '초기 대기'
        previous = ('STAGE 1', '관측 자세로 이동')
        for run in self.runs:
            if seconds < run['start_relative_walk_s']:
                return previous
            previous = (f'STAGE {run["stage"]}', run['state'])
            if seconds <= run['end_relative_walk_s']:
                events = self.rows[run['run_id']]
                row = next((r for r in reversed(events) if r['monotonic_s'] <= seconds + self.origin), events[0])
                return f'STAGE {run["stage"]}', row.get('phase') or row.get('state', '')
        return previous
