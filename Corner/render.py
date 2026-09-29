"""검증된 전환 경로를 네 개의 외부 시점에서 동시에 보여준다.
영상·미리보기·주요 장면을 Corner 출력 폴더에만 저장한다.
"""

import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Corner.scene import OUTPUT, Scene
from Corner.trajectory import Trajectory
from Corner.validate import digest

INK = (24, 42, 60)
MUTED = (96, 112, 129)
FRAME_SIZE = (1920, 1080)
PANEL_SIZE = (948, 500)
TRAIL_COLORS = {"tip_L": (35, 115, 230), "tip_R": (240, 115, 30)}


def font(size):
    """사용 가능한 한글 글꼴을 찾아 영상 자막에 사용한다."""
    paths = [Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
             Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("한글 자막용 AppleSDGothicNeo 또는 NotoSansCJK 글꼴이 필요합니다.")


class Composer:
    """동일한 로봇 상태를 전체·정면·측면·상단 화면에 배치한다."""

    def __init__(self, scene, trajectory):
        """외부 고정 시점과 자막 글꼴을 준비하며 센서 카메라는 사용하지 않는다."""
        self.scene = scene
        self.trajectory = trajectory
        # 영상 배경을 단순하게 보이도록 바닥 표시를 숨긴다.
        scene.model.geom_rgba[scene.model.geom("floor").id, 3] = 0.
        self.renderer = mujoco.Renderer(scene.model, height=PANEL_SIZE[1], width=PANEL_SIZE[0])
        self.views = []
        definitions = [("OVERVIEW", 132, -18, .74, False, (8, 8)),
                       ("FRONT", 90, -3, .68, True, (964, 8)),
                       ("SIDE", 300, -20, .68, True, (8, 516)),
                       ("TOP", 90, -90, .72, True, (964, 516))]
        for label, azimuth, elevation, distance, orthographic, origin in definitions:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.095, .105, .175]
            if label in ("FRONT", "SIDE"):
                camera.lookat[2] = .155
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.orthographic = orthographic
            self.views.append((label, camera, origin))
        self.fonts = {size: font(size) for size in (19, 23, 27, 28)}
        self.prepare_trails()

    def prepare_trails(self):
        """실제 관절 보간과 고정단을 적용해 두 중심 사이트의 월드 궤적을 미리 계산한다."""
        self.trail_times = np.unique(np.r_[np.arange(0., self.trajectory.duration, 1 / 60),
                                           self.trajectory.arrays["time"]])
        self.trail_positions = {name: np.empty((len(self.trail_times), 3)) for name in TRAIL_COLORS}
        for index, time in enumerate(self.trail_times):
            q, grips, anchor, _ = self.trajectory.sample(time)
            self.scene.set(q, grips, anchor)
            for name, points in self.trail_positions.items():
                points[index] = self.scene.data.site(name).xpos

    def top_pixels(self, points):
        r"""현재 상단 직교 카메라로 월드 궤적을 패널 픽셀 좌표에 투영한다.

        $$u=\frac{W}{2}+\frac{H}{h}(r^T(p-c)-f_c),\quad
        v=\frac{H}{h}(f_t-a^T(p-c))$$

        c는 카메라 위치, r과 a는 화면 오른쪽·위 방향이다.
        W와 H는 패널 크기, h는 시야 높이, f_c와 f_t는 시야 중심·상단 좌표다.
        """
        cameras = self.renderer.scene.camera
        camera = mujoco.mjv_averageCamera(cameras[0], cameras[1])
        if not camera.orthographic:
            raise ValueError("궤적 투영에는 상단 직교 카메라가 필요합니다.")
        # 카메라 기준 위치 벡터: $$d=p-c$$
        relative = np.asarray(points) - camera.pos
        # 화면 오른쪽 방향: $$r=f\times a$$
        right = np.cross(camera.forward, camera.up)
        # 상하 시야의 실제 높이: $$h=f_t-f_b$$
        height = camera.frustum_top - camera.frustum_bottom
        # 월드 길이당 픽셀 수: $$s=H/h$$
        scale = PANEL_SIZE[1] / height
        # 화면 가로 좌표: $$u=W/2+s(r^Td-f_c)$$
        horizontal = PANEL_SIZE[0] / 2 + scale * (relative @ right - camera.frustum_center)
        # 화면 세로 좌표: $$v=s(f_t-a^Td)$$
        vertical = scale * (camera.frustum_top - relative @ camera.up)
        return np.column_stack((horizontal, vertical))

    def draw_trails(self, panel, time):
        """현재 시각까지의 두 사이트 궤적과 현재 위치를 서로 다른 색으로 덧그린다."""
        draw = ImageDraw.Draw(panel)
        end = np.searchsorted(self.trail_times, time, side="right")
        for name, color in TRAIL_COLORS.items():
            points = np.vstack((self.trail_positions[name][:end], self.scene.data.site(name).xpos))
            pixels = [tuple(point) for point in self.top_pixels(points)]
            if len(pixels) > 1:
                draw.line(pixels, fill="white", width=6, joint="curve")
                draw.line(pixels, fill=color, width=3, joint="curve")
            x, y = pixels[-1]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=2)

    def panel(self, label, camera):
        """외부 시점을 렌더링하고 상단 시점에서만 빔을 반투명으로 표시한다."""
        model = self.scene.model
        colors = model.geom_rgba[self.scene.beams].copy()
        projection = (model.vis.global_.orthographic, model.vis.global_.fovy)
        try:
            model.vis.global_.orthographic = camera.orthographic
            model.vis.global_.fovy = (.43 if label == "TOP" else .48) if camera.orthographic else 45.
            if label == "TOP":
                model.geom_rgba[self.scene.beams, 3] = .20
            self.renderer.update_scene(self.scene.data, camera=camera)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = label != "TOP"
            return Image.fromarray(self.renderer.render())
        finally:
            model.geom_rgba[self.scene.beams] = colors
            model.vis.global_.orthographic, model.vis.global_.fovy = projection

    def frame(self, time):
        """한 시각의 네 화면과 시점 이름·재생 시각·현재 단계를 합성한다."""
        q, grips, anchor, phase = self.trajectory.sample(time)
        self.scene.set(q, grips, anchor)
        frame = Image.new("RGB", FRAME_SIZE, (238, 243, 247))
        draw = ImageDraw.Draw(frame)
        for label, camera, (x, y) in self.views:
            panel = self.panel(label, camera)
            if label == "TOP":
                self.draw_trails(panel, time)
            frame.paste(panel, (x, y))
            width = draw.textlength(label, font=self.fonts[28])
            center = x + PANEL_SIZE[0] / 2
            baseline = y + PANEL_SIZE[1] - 38
            draw.rectangle((center - width / 2 - 9, baseline - 3,
                            center + width / 2 + 9, baseline + 30), fill=(245, 248, 250))
            draw.text((center, baseline), label, anchor="mt", font=self.fonts[28], fill=INK)
        caption = f"{time:05.2f} / {self.trajectory.duration:05.2f} s   |   {self.trajectory.meta['phases'][phase]}"
        draw.text((FRAME_SIZE[0] / 2, 1037), caption, anchor="mt", font=self.fonts[27], fill=INK)
        return frame

    def close(self):
        """렌더링 자원을 해제한다."""
        self.renderer.close()


def render(directory=OUTPUT, fps=30):
    """검사에 통과한 현재 경로를 H.264 영상과 주요 장면 이미지로 저장한다."""
    if fps < 1 or fps > 60:
        raise ValueError("영상 프레임 수는 1부터 60 사이여야 합니다.")
    check = json.loads((directory / "validation.json").read_text())
    if (not check["passed"] or check["trajectory_sha256"] != digest(directory / "trajectory.npz")
            or check["scene_sha256"] != digest(directory / "scene/model.xml")):
        raise ValueError("현재 경로와 장면의 충돌 검사를 먼저 통과해야 합니다.")
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise RuntimeError("MP4 영상 생성에 ffmpeg가 필요합니다.")
    trajectory = Trajectory(directory)
    composer = Composer(Scene(directory / "scene/model.xml"), trajectory)
    path = directory / "corner_transfer_4views.mp4"
    temporary = directory / "corner_transfer_4views.partial.mp4"
    command = [encoder, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1920x1080", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        total = int(np.ceil(trajectory.duration * fps)) + 1
        for index in range(total):
            time = min(index / fps, trajectory.duration)
            frame = composer.frame(time)
            process.stdin.write(frame.tobytes())
            if index % (fps * 10) == 0:
                print(f"영상: {time:.1f}/{trajectory.duration:.1f}초", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg 영상 인코딩이 실패했습니다.")
        temporary.replace(path)
        stages = (0, 4, 7, 11, 13, len(trajectory.meta["phases"]) - 1)
        stills = []
        for order, phase in enumerate(stages):
            indices = np.flatnonzero(trajectory.arrays["phase"] == phase)
            time = float(trajectory.arrays["time"][indices[-1]])
            frame = composer.frame(time)
            frame.save(directory / f"four_views_keyframe_{order + 1:02d}.png")
            stills.append(frame.resize((640, 360), Image.Resampling.LANCZOS))
        sheet = Image.new("RGB", (1920, 720), "white")
        for index, frame in enumerate(stills):
            sheet.paste(frame, ((index % 3) * 640, (index // 3) * 360))
        sheet.save(directory / "four_views_contact_sheet.png")
    finally:
        composer.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    print(f"영상 저장: {path}", flush=True)
    return path
