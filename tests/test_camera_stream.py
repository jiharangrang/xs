"""독립 소비자의 새 프레임 수신과 단일 카메라 연결·해제·실패 복구를 검증한다."""

from threading import Event
import time
import unittest

import numpy as np

from hardware.camera import CameraError, RGBDFrame
from hardware.camera_stream import CameraStream


class FakeCamera:
    """작은 원본 프레임을 일정 간격으로 생성하는 카메라 대역이다."""

    def __init__(self, factory):
        """연결 횟수 기록과 오류 주입 상태를 준비한다."""
        self.factory = factory
        self.info = {"aligned": False, "depth_unit": "m"}
        self.failed = Event()
        self.sequence = 0

    def __enter__(self):
        """카메라 연결 횟수를 기록한다."""
        self.factory.opened += 1
        return self

    def read(self, *, timeout_ms=5000):
        """새 프레임마다 다른 시각을 부여하고 요청된 수신 실패를 재현한다."""
        if self.failed.wait(.005):
            raise CameraError("시험용 장치 분리")
        self.sequence += 1
        return RGBDFrame(np.full((18, 32, 3), self.sequence % 255, np.uint8),
                         np.full((16, 24), .25, np.float32), self.sequence, self.sequence,
                         time.monotonic())

    def __exit__(self, *args):
        """카메라 종료 횟수를 기록한다."""
        self.factory.closed += 1


class FakeCameraFactory:
    """시험용 연결마다 별도의 카메라 객체를 제공한다."""

    def __init__(self):
        """연결·종료 횟수를 초기화한다."""
        self.opened = self.closed = 0
        self.latest = None

    def __call__(self):
        """가장 최근에 생성한 카메라를 오류 주입용으로 보관한다."""
        self.latest = FakeCamera(self)
        return self.latest


class CameraStreamTests(unittest.TestCase):
    """표시와 스캔의 수신 순서가 서로 영향을 주지 않는지 확인한다."""

    def test_readers_share_one_camera_and_wait_for_independent_new_frames(self):
        """여러 소비자가 장치를 한 번만 열고 마지막 소비자가 종료할 때 닫는다."""
        factory = FakeCameraFactory()
        stream = CameraStream(factory)
        with stream.reader() as scan:
            with stream.reader() as preview:
                first = preview.read()
                second = preview.read()
                self.assertGreater(second.depth_timestamp_us, first.depth_timestamp_us)
                self.assertGreaterEqual(scan.read().depth_timestamp_us, second.depth_timestamp_us)
                self.assertEqual(factory.opened, 1)
            self.assertEqual(factory.closed, 0)
            self.assertIsNotNone(scan.read())
        self.assertEqual(factory.closed, 1)
        with self.assertRaises(CameraError):
            scan.read()

    def test_source_failure_reaches_consumers_and_new_connection_recovers(self):
        """장치 오류를 오래된 프레임으로 숨기지 않고 새 연결에서는 복구할 수 있다."""
        factory = FakeCameraFactory()
        stream = CameraStream(factory)
        with stream.reader() as reader:
            reader.read()
            factory.latest.failed.set()
            with stream._condition:
                self.assertTrue(stream._condition.wait_for(lambda: stream._error is not None, timeout=1))
            with self.assertRaisesRegex(CameraError, "장치 분리"):
                reader.read()
        with stream.reader() as reader:
            self.assertIsNotNone(reader.read())
        self.assertEqual((factory.opened, factory.closed), (2, 2))


if __name__ == "__main__":
    unittest.main()
