"""측정한 경로 준비 시간과 저장된 관절 경로를 좌우 비교 영상으로 재생한다.
실측 계산 시간과 시뮬레이션 이동 시간을 구분하며 실물 장치에는 연결하지 않는다.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from presentation.rendering import set_camera, setup_scene

REPO = Path(__file__).resolve().parents[1]
FRAME_SIZE = (1920, 1080)
PANEL_SIZE = (960, 1080)
PANEL_ORIGINS = ((0, 0), (960, 0))
BACKGROUND = (235, 242, 247)
INK = (28, 45, 62)
MUTED = (91, 109, 126)
RED = (218, 58, 76)
GREEN = (24, 163, 99)
JOINTS = tuple(f'J{i}' for i in range(1, 8)) + ('G_R',)


def digest(path):
    """영상의 근거 파일과 현재 모델의 식별자를 계산한다."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size, *, bold=False, numeric=False):
    """한글 설명과 고정 폭 타이머에 맞는 글꼴을 준비한다."""
    if numeric:
        name = 'Menlo.ttc'
    elif bold:
        name = 'Supplemental/Arial Bold.ttf'
    else:
        name = 'AppleSDGothicNeo.ttc'
    path = Path('/System/Library/Fonts') / name
    return ImageFont.truetype(str(path), size)


@dataclass
class Method:
    """한 방법의 실제 계산 시간과 시뮬레이션 경로를 보관한다."""

    name: str
    title: str
    q: np.ndarray
    times: np.ndarray
    stages: np.ndarray
    planning_s: float
    ready_s: float

    @property
    def arrival_s(self):
        r"""계산 준비와 시뮬레이션 이동을 합한 도착 시각을 반환한다.

        $$t_{arrival}=t_{ready}+T_{motion}$$
        """
        # 공통 시작점에서 본 도착 시각: $$t_{arrival}=t_{ready}+T_{motion}$$
        return self.ready_s + float(self.times[-1])

    def sample(self, elapsed):
        r"""계산 중에는 초기 자세를 유지하고 준비 이후 저장 경로를 보간한다.

        $$q(t)=\operatorname{interp}(\operatorname{clip}(t-t_{ready},0,T),\{t_i,q_i\})$$
        """
        # 경로 안에서 경과한 이동 시간: $$\tau=\operatorname{clip}(t-t_{ready},0,T)$$
        local = float(np.clip(elapsed - self.ready_s, 0., self.times[-1]))
        q = np.array([np.interp(local, self.times, self.q[:, column]) for column in range(self.q.shape[1])])
        if elapsed < self.planning_s:
            state = '경로 계산 중'
        elif elapsed < self.ready_s:
            state = '최종 메쉬 확인 중'
        elif elapsed < self.arrival_s:
            state = '이동 중'
        else:
            state = '도착 완료'
        return q, state, elapsed >= self.arrival_s


def load_comparison(directory, scene_directory):
    """검증된 두 경로와 중앙값 계산 시간을 읽고 모델 식별자를 대조한다."""
    directory, scene_directory = Path(directory), Path(scene_directory)
    report = json.loads((directory / 'results.json').read_text())
    scene = json.loads((scene_directory / 'scene.json').read_text())
    if not report['all_paths_valid'] or report['scene_id'] != scene['scene_id']:
        raise ValueError('검사에 통과한 동일 장면의 비교 결과가 필요합니다.')
    for name, key in [('model.xml', 'model_sha256'), ('camera_frames.xml', 'camera_frames_sha256'),
                      ('calibration.yaml', 'calibration_sha256')]:
        if digest(REPO / 'models/xs' / name) != scene[key]:
            raise ValueError(f'측정 이후 모델이 바뀌었습니다: {name}')
    methods = []
    for name, title in [('mesh', 'MESH COLLISION'), ('learned', 'LEARNED MODEL')]:
        with np.load(directory / f'{name}_path.npz', allow_pickle=False) as data:
            q, times, stages = (data[key].copy() for key in ('q', 'time_s', 'stage'))
        if (q.shape != (len(times), len(JOINTS)) or not np.isfinite(q).all()
                or times[0] != 0 or np.any(np.diff(times) <= 0)):
            raise ValueError('저장 경로의 관절각 또는 시각 형식이 올바르지 않습니다.')
        summary = report['summary'][name]
        methods.append(Method(name, title, q, times, stages,
                              summary['planning_s']['median'], summary['total_s']['median']))
    if not np.array_equal(methods[0].times, methods[1].times):
        raise ValueError('두 방식의 시뮬레이션 이동 시간은 같아야 합니다.')
    return methods, report, scene


class Composer:
    """같은 고정 카메라의 두 화면과 상태 테두리·공통 타이머를 합성한다."""

    def __init__(self, methods, scene, speed):
        """발표용 색감과 동일 시야를 준비하고 경로의 렌더링 캐시를 만든다."""
        self.methods, self.speed = methods, speed
        source = SimpleNamespace(recording=SimpleNamespace(model_path=REPO / 'models/xs/model.xml'))
        self.model, self.data, self.options = setup_scene(source, *PANEL_SIZE)
        self.model.body('ibeam').pos[:] = scene['beam_pose']['pos']
        self.model.body('ibeam').quat[:] = scene['beam_pose']['quat']
        self.options.geomgroup[3:] = 0
        self.indices = [self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINTS]
        self.camera = mujoco.MjvCamera()
        set_camera(self.camera, 'side')
        # 기존 SIDE 방향을 공유하고 세로 화면에서 초기 자세 전체가 보이도록 중심과 거리만 맞춘다.
        self.camera.lookat[:] = [.09, -.03636, -.02]
        self.camera.distance = .90
        self.renderer = mujoco.Renderer(self.model, height=PANEL_SIZE[1], width=PANEL_SIZE[0])
        self.fonts = {size: font(size) for size in (24, 26)}
        self.title_font = font(34)
        self.title_width = max(self.title_font.getlength(method.title) for method in methods) + 48
        self.seconds_font = font(86, numeric=True)
        self.centis_font = font(36, numeric=True)
        self.cache = {}
        self.glows = {False: self.glow(RED), True: self.glow(GREEN)}

    def glow(self, color):
        r"""화면 경계에서 안쪽으로 옅어지는 상태 테두리를 만든다.

        $$\alpha(d)=0.72\exp(-(d/15)^2)$$
        """
        width, height = PANEL_SIZE
        y, x = np.mgrid[:height, :width]
        edge = np.minimum.reduce((x, y, width - 1 - x, height - 1 - y))
        # 경계에서 안쪽으로 감소하는 불투명도: $$\alpha(d)=0.72\exp(-(d/15)^2)$$
        alpha = .72 * np.exp(-(edge / 15.) ** 2)
        alpha[edge < 4] = .95
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        pixels[:, :, :3] = color
        pixels[:, :, 3] = np.round(alpha * 255).astype(np.uint8)
        return Image.fromarray(pixels)

    def robot(self, q):
        """물리 시간을 진행하지 않고 저장된 관절각의 영상만 렌더링한다."""
        key = q.tobytes()
        if key not in self.cache:
            self.data.qpos[self.indices] = q
            self.data.joint('G_L').qpos[0] = 0.
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
            panel = Image.fromarray(self.renderer.render()).convert('RGBA')
            if len(self.cache) > 6:
                self.cache.clear()
            self.cache[key] = panel
        return self.cache[key].copy()

    def frame(self, elapsed):
        """화면을 채운 두 영상 위에 제목·진행 상태·숫자 타이머만 표시한다."""
        frame = Image.new('RGB', FRAME_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(frame)
        for method, origin in zip(self.methods, PANEL_ORIGINS, strict=True):
            q, state, arrived = method.sample(elapsed)
            color = GREEN if arrived else RED
            panel = self.robot(q)
            panel.alpha_composite(self.glows[arrived])
            local = ImageDraw.Draw(panel)
            label = '✓  도착 완료' if arrived else '이동 중' if state == '이동 중' else '계산 중'
            text_width = local.textlength(label, font=self.fonts[26])
            left = (PANEL_SIZE[0] - text_width) / 2
            local.rounded_rectangle((left - 20, 1004, left + text_width + 20, 1048), radius=18,
                                    fill=(*BACKGROUND, 246), outline=(*color, 255), width=2)
            local.text((PANEL_SIZE[0] / 2, 1026), label, anchor='mm', font=self.fonts[26], fill=color)
            center = PANEL_SIZE[0] / 2
            local.rectangle((center - self.title_width / 2, 22,
                             center + self.title_width / 2, 82),
                            fill=(*BACKGROUND, 255), outline=(192, 209, 222, 255), width=1)
            local.text((center, 52), method.title, anchor='mm', font=self.title_font, fill=INK)
            frame.paste(panel.convert('RGB'), origin)
        draw.rounded_rectangle((844, 906, 1076, 1068), radius=24, fill=BACKGROUND)
        ticks = round(max(0., elapsed) * 100)
        seconds, centis = divmod(ticks, 100)
        draw.text((960, 966), f'{seconds:02}', anchor='mm', font=self.seconds_font, fill=INK)
        draw.text((1041, 986), '초', anchor='mm', font=self.fonts[24], fill=MUTED)
        draw.text((960, 1034), f'.{centis:02}', anchor='mm', font=self.centis_font, fill=MUTED)
        return frame

    def close(self):
        """그래픽 자원을 닫는다."""
        self.renderer.close()


def render(directory, scene_directory, output, *, speed=2., fps=30, preview=False):
    r"""실측 대기와 동일 이동을 실제 시간축으로 합성한 영상을 저장한다.

    $$t_k=\min(kv/f,\max_j t_{arrival,j})$$

    마지막 도착 후에는 타이머를 정지하고 결과 화면을 잠깐 유지한다.
    """
    output = Path(output).resolve()
    if not np.isfinite(speed) or speed <= 0 or not 1 <= fps <= 60:
        raise ValueError('배속과 프레임률이 유효하지 않습니다.')
    if output.exists() and not preview:
        raise FileExistsError(f'기존 영상은 덮어쓰지 않습니다: {output}')
    methods, report, scene = load_comparison(directory, scene_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_arrival = max(method.arrival_s for method in methods)
    composer = Composer(methods, scene, speed)
    try:
        samples = [0., 6., methods[1].arrival_s + .2, last_arrival]
        if preview:
            for index, elapsed in enumerate(samples):
                composer.frame(elapsed).save(output.with_name(f'{output.stem}_preview_{index}.jpg'))
            return
        # 두 배속과 종료 뒤 삼 초의 실제 시간 여유를 반영한 프레임 수: $$N=\lceil(T_{last}+3)f/v\rceil$$
        count = math.ceil((last_arrival + 3) * fps / speed)
        with tempfile.TemporaryDirectory(prefix='xs-planning-', dir=output.parent) as temporary:
            destination = Path(temporary) / 'render.mp4'
            command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', '1920x1080', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                       '-threads', '4', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart', str(destination)]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                try:
                    for index in range(count):
                        # 배속과 종료 후 정지를 반영한 실제 경과 시간: $$t_k=\min(kv/f,T_{last})$$
                        elapsed = min(index * speed / fps, last_arrival)
                        process.stdin.write(np.asarray(composer.frame(elapsed)).tobytes())
                        if index % (fps * 3) == 0:
                            print(f'계산 비교 영상 {index / count:.0%}', flush=True)
                    process.stdin.close()
                    if process.wait():
                        errors.seek(0)
                        raise RuntimeError(errors.read().decode())
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
            destination.replace(output)
        metadata = {'benchmark_results': str(Path(directory).resolve() / 'results.json'),
                    'benchmark_sha256': digest(Path(directory) / 'results.json'),
                    'fps': fps, 'frames': count, 'playback_speed': speed,
                    'layout': 'full_height_minimal', 'panel_size_px': list(PANEL_SIZE),
                    'title_style': {'font': composer.title_font.getname(), 'size_px': 34,
                                    'background': 'square_corners_equal_width'},
                    'movement_time_basis': 'saved_simulation_time_s_not_physical_measurement',
                    'goal': '20cm_inserted_right_jaw_open', 'timing_statistic': 'five_run_median_with_final_mesh_validation',
                    'methods': {method.name: {'ready_s': method.ready_s, 'motion_s': float(method.times[-1]),
                                              'arrival_s': method.arrival_s,
                                              'path_sha256': digest(Path(directory) / f'{method.name}_path.npz')}
                                for method in methods},
                    'camera': {'preset': 'side', 'azimuth': composer.camera.azimuth, 'elevation': composer.camera.elevation,
                               'distance': composer.camera.distance, 'lookat': composer.camera.lookat.tolist()}}
        output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(f'저장 완료: {output}', flush=True)
    finally:
        composer.close()


def main():
    """기존 비교 결과를 읽어 별도의 발표 영상을 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=REPO / 'Learning/artifacts/linux_path_planning_20cm')
    parser.add_argument('--scene', type=Path, default=REPO / 'Learning/artifacts/scene')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/presentation/planning_comparison/mesh_vs_learned_side_modern_2x.mp4')
    parser.add_argument('--speed', type=float, default=2.)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    render(args.benchmark, args.scene, args.output, speed=args.speed, fps=args.fps, preview=args.preview)


if __name__ == '__main__':
    main()
