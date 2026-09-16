"""공유 RGB-D 수신을 화면용 JPEG와 상태 API로 연결한다.
원본 깊이와 모터 실행은 변경하지 않으며 영상 인코딩은 별도 스레드에서 수행한다.
"""

import asyncio
from threading import Event, RLock, Thread
import time

from fastapi import HTTPException
from fastapi.responses import Response
import numpy as np

from hardware.camera import CameraError
from hardware.camera_stream import CameraStream


def preview_images(frame) -> dict[str, bytes]:
    r"""RGB와 고정 거리 범위의 깊이 표시를 작은 JPEG 영상으로 변환한다.

    $$u=\operatorname{clip}((d-0.10)/0.60,0,1)$$

    d는 원본 깊이이며 무효 깊이는 검정으로 표시한다.
    """
    import cv2

    color = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
    depth = np.where(valid, frame.depth_m, .7)
    # 화면용 거리 정규화: $$u=\operatorname{clip}((d-0.10)/0.60,0,1)$$
    normalized = np.clip((depth - .1) / .6, 0, 1)
    # 가까운 깊이가 높은 색상표 값을 갖도록 변환: $$c=255(1-u)$$
    colors = np.round(255 * (1 - normalized)).astype(np.uint8)
    depth_color = cv2.applyColorMap(colors, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    images = {}
    for name, source in (("color", color), ("depth", depth_color)):
        height, width = source.shape[:2]
        if width > 640:
            # 원본 종횡비를 유지하는 표시 높이: $$h'=\operatorname{round}(640h/w)$$
            display_height = round(height * 640 / width)
            source = cv2.resize(source, (640, display_height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            raise CameraError("실시간 영상을 압축하지 못했습니다.")
        images[name] = encoded.tobytes()
    return images


class CameraPreview:
    """최신 표시 영상만 유지하며 오류가 발생하면 마지막 영상을 실시간으로 제공하지 않는다."""

    def __init__(self, stream=None, *, encoder=preview_images) -> None:
        """카메라를 열지 않고 공용 스트림과 표시 상태를 준비한다."""
        self.stream = stream if stream is not None else CameraStream()
        self._encoder = encoder
        self._lock = RLock()
        self._lifecycle = RLock()
        self._thread = None
        self._stop = Event()
        self._state = "STOPPED"
        self._message = "카메라 연결을 기다립니다."
        self._images = {}
        self._received_at = None
        self._sequence = 0

    def start(self) -> dict:
        """기존 표시 작업이 없을 때만 카메라 수신과 인코딩을 시작한다."""
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            with self._lock:
                self._stop = Event()
                self._state = "STARTING"
                self._message = "카메라 연결 중…"
                self._images = {}
                self._received_at = None
            self._thread = Thread(target=self._run, name="camera-preview", daemon=True)
            self._thread.start()
            return self.status()

    def _run(self) -> None:
        """원본 수신 소비자 하나를 유지하고 모터 조회와 독립적으로 영상을 압축한다."""
        try:
            with self.stream.reader() as camera:
                while not self._stop.is_set():
                    frame = camera.read(timeout_ms=5000 if self._received_at is None else 2500)
                    images = self._encoder(frame)
                    with self._lock:
                        if self._stop.is_set():
                            break
                        self._images = images
                        self._received_at = frame.received_monotonic_s
                        self._sequence += 1
                        self._state = "LIVE"
                        self._message = "실시간 수신 중"
        except Exception as error:
            with self._lock:
                if not self._stop.is_set():
                    self._state = "ERROR"
                    self._message = str(error)
                    if "uvc_open" in str(error) and "-3" in str(error):
                        self._message = "카메라 USB 접근 권한이 없습니다. sudo로 실행한 서버가 필요합니다."
                self._images = {}

    def status(self) -> dict:
        """연결 상태와 최신 프레임 경과 시간을 반환한다."""
        with self._lock:
            age = None if self._received_at is None else time.monotonic() - self._received_at
            state = "STALLED" if self._state == "LIVE" and age > 1.5 else self._state
            return {"state": state, "message": "새 영상을 기다립니다." if state == "STALLED" else self._message,
                    "sequence": self._sequence, "age_s": age}

    def image(self, mode: str) -> tuple[bytes, dict]:
        """신선한 표시 영상만 반환하며 연결 오류나 오래된 프레임은 거부한다."""
        with self._lock:
            status = self.status()
            if status["state"] != "LIVE" or mode not in self._images:
                raise CameraError(status["message"])
            return self._images[mode], status

    def stop(self) -> dict:
        """표시 소비자만 종료하며 진행 중인 스캔의 공유 카메라는 유지한다."""
        with self._lifecycle:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=9.)
                if self._thread.is_alive():
                    raise CameraError("실시간 카메라 표시를 종료 중입니다.")
            with self._lock:
                self._state = "STOPPED"
                self._message = "카메라 표시가 꺼져 있습니다."
                self._images = {}
                self._received_at = None
            return self.status()


def install_routes(app, console) -> None:
    """카메라 시작·중지·상태·표시 영상 경로를 등록한다."""
    @app.get("/api/camera")
    async def status():
        """현재 카메라 표시 상태를 조회한다."""
        return console().camera.status()

    @app.post("/api/camera/start")
    async def start():
        """모터 명령 없이 카메라 표시를 시작한다."""
        return await asyncio.to_thread(console().camera.start)

    @app.post("/api/camera/stop")
    async def stop():
        """화면용 카메라 소비자를 종료한다."""
        try:
            return await asyncio.to_thread(console().camera.stop)
        except CameraError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/camera/frame/{mode}")
    async def frame(mode: str):
        """RGB 또는 깊이 표시 JPEG를 브라우저 캐시 없이 전달한다."""
        if mode not in ("color", "depth"):
            raise HTTPException(status_code=404, detail="지원하지 않는 카메라 영상입니다.")
        try:
            data, status = console().camera.image(mode)
        except CameraError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return Response(data, media_type="image/jpeg", headers={
            "Cache-Control": "no-store", "X-Frame-Sequence": str(status["sequence"]),
        })
