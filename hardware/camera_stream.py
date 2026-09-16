"""카메라를 한 번만 열고 최신 RGB·깊이 프레임을 여러 소비자에게 전달한다.
각 소비자는 독립적인 수신 순서를 가지며 마지막 소비자가 나가면 장치를 닫는다.
"""

from threading import Condition, Event, RLock, Thread
import time

from hardware.camera import CameraError, Gemini215Camera


class CameraStream:
    """실시간 표시와 원본 스캔이 같은 장치와 최신 프레임을 공유하도록 관리한다."""

    def __init__(self, camera_factory=Gemini215Camera) -> None:
        """장치를 열지 않고 수신 상태와 소비자 수를 준비한다."""
        self._factory = camera_factory
        self._lifecycle = RLock()
        self._condition = Condition()
        self._users = 0
        self._thread = None
        self._stop = Event()
        self._latest = None
        self._sequence = 0
        self._info = {}
        self._error = None

    def reader(self):
        """독립적인 프레임 순서를 사용하는 문맥 관리자를 만든다."""
        return CameraReader(self)

    def _acquire(self) -> None:
        """첫 소비자에게만 새 수신 작업을 열고 나머지는 같은 작업에 연결한다."""
        with self._lifecycle:
            if self._users:
                if self._error is not None:
                    raise CameraError(str(self._error))
                self._users += 1
                return
            if self._thread is not None and self._thread.is_alive():
                raise CameraError("이전 카메라 연결을 종료 중입니다. 잠시 후 다시 연결해 주세요.")
            with self._condition:
                self._stop = Event()
                self._latest = None
                self._sequence = 0
                self._info = {}
                self._error = None
            self._users = 1
            self._thread = Thread(target=self._receive, name="camera-receiver", daemon=True)
            self._thread.start()

    def _receive(self) -> None:
        """SDK 연결과 수신·종료를 한 스레드에서 수행하고 최신 원본만 보관한다."""
        try:
            with self._factory() as camera:
                while not self._stop.is_set():
                    frame = camera.read(timeout_ms=1000)
                    with self._condition:
                        self._latest = frame
                        self._info = camera.info
                        self._sequence += 1
                        self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._latest = None
                self._error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._condition.notify_all()

    def _read(self, previous: int, timeout_ms: int):
        """소비자가 아직 읽지 않은 새 프레임을 기다리고 수신 오류를 그대로 전달한다."""
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms는 양의 정수여야 합니다.")
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while True:
                if self._stop.is_set():
                    raise CameraError("카메라 수신이 종료되었습니다.")
                if self._error is not None:
                    raise CameraError(str(self._error)) from self._error
                if self._latest is not None and self._sequence > previous:
                    return self._sequence, self._latest, self._info
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError("제한 시간 안에 새 카메라 영상을 받지 못했습니다.")
                self._condition.wait(remaining)

    def _release(self) -> None:
        """마지막 소비자가 떠나면 수신 작업의 종료를 기다려 장치 중복 연결을 막는다."""
        with self._lifecycle:
            self._users -= 1
            if self._users:
                return
            self._stop.set()
            with self._condition:
                self._condition.notify_all()
            self._thread.join(timeout=6.)
            if self._thread.is_alive():
                raise CameraError("카메라 종료가 지연되고 있습니다.")
            with self._condition:
                self._latest = None


class CameraReader:
    """공유 수신 작업에서 자기 순서보다 새로운 프레임을 읽는 소비자다."""

    def __init__(self, stream: CameraStream) -> None:
        """연결 전 읽기 순서와 카메라 정보를 초기화한다."""
        self._stream = stream
        self._sequence = 0
        self._active = False
        self.info = {}

    def __enter__(self):
        """장치를 공유하고 수신을 시작한다."""
        if self._active:
            raise CameraError("이미 사용 중인 카메라 읽기 객체입니다.")
        self._stream._acquire()
        self._active = True
        self._sequence = 0
        return self

    def read(self, *, timeout_ms: int = 5000):
        """현재 소비자에게 새로운 원본 프레임과 장치 정보를 전달한다."""
        if not self._active:
            raise CameraError("먼저 카메라 읽기를 시작해 주세요.")
        self._sequence, frame, self.info = self._stream._read(self._sequence, timeout_ms)
        return frame

    def __exit__(self, *args) -> None:
        """현재 소비자의 연결만 반환한다."""
        if self._active:
            self._active = False
            self._stream._release()
