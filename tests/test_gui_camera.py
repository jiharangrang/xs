"""카메라 표시와 원본 스캔의 동시 사용 및 JPEG·오래된 영상 처리를 검증한다."""

from pathlib import Path
from threading import Event
import tempfile
import time
import unittest

import numpy as np

from gui.camera import CameraPreview, preview_images
from hardware.camera import CameraError, RGBDFrame
from hardware.camera_scan import capture_scan
from hardware.camera_stream import CameraStream
from test_camera_stream import FakeCameraFactory


class CameraPreviewTests(unittest.TestCase):
    """모터나 실제 카메라 없이 표시 소비자의 수명과 스캔 공유를 확인한다."""

    def setUp(self):
        """하나의 원본 스트림에 화면 인코더를 연결한다."""
        self.factory = FakeCameraFactory()
        self.stream = CameraStream(self.factory)
        self.encoded = Event()

        def encode(frame):
            """원본 프레임 도착을 알리고 작은 표시 데이터로 바꾼다."""
            self.encoded.set()
            return {"color": b"color-jpeg", "depth": b"depth-jpeg"}

        self.preview = CameraPreview(self.stream, encoder=encode)
        self.addCleanup(self.preview.stop)

    def _wait_live(self):
        """표시 상태가 실제 수신 완료로 바뀔 때까지 제한된 시간 안에서 기다린다."""
        self.assertTrue(self.encoded.wait(1))
        deadline = time.monotonic() + 1
        while self.preview.status()["state"] != "LIVE" and time.monotonic() < deadline:
            Event().wait(.001)
        self.assertEqual(self.preview.status()["state"], "LIVE")

    def test_preview_stop_keeps_scan_reader_and_restart_reuses_source(self):
        """표시를 꺼도 별도 스캔 소비자는 계속 수신하며 재연결이 장치를 중복 개방하지 않는다."""
        with self.stream.reader() as scan:
            self.preview.start()
            self.preview.start()
            self._wait_live()
            self.assertEqual(self.preview.image("color")[0], b"color-jpeg")
            self.preview.stop()
            self.assertIsNotNone(scan.read())
            self.assertEqual(self.factory.closed, 0)
            self.preview.start()
            self._wait_live()
            self.assertEqual(self.factory.opened, 1)
            self.preview.stop()
        self.assertEqual(self.factory.closed, 1)

    def test_stale_or_failed_input_is_not_reported_as_live(self):
        """오래된 영상과 장치 분리 이후의 마지막 영상을 정상 스트림으로 제공하지 않는다."""
        self.preview.start()
        self._wait_live()
        with self.preview._lock:
            self.preview._received_at = time.monotonic() - 2
            self.assertEqual(self.preview.status()["state"], "STALLED")
            with self.assertRaises(CameraError):
                self.preview.image("color")
        self.factory.latest.failed.set()
        deadline = time.monotonic() + 1
        while self.preview.status()["state"] != "ERROR" and time.monotonic() < deadline:
            Event().wait(.005)
        self.assertEqual(self.preview.status()["state"], "ERROR")
        with self.assertRaises(CameraError):
            self.preview.image("depth")

    def test_original_scan_and_preview_use_single_device(self):
        """실시간 표시 중 스캔해도 단일 장치에서 원본 프레임 묶음을 저장한다."""
        self.preview.start()
        self._wait_live()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "motors.jsonl"
            log.touch()
            result = capture_scan(lambda: {"states": [{"name": "J1", "speed_deg_s": 0}]},
                                  log, root / "scans", camera_factory=self.stream.reader)
            self.assertEqual(result["frames"], 15)
            self.assertTrue((Path(result["path"]) / "depth_frames.npz").exists())
            self.assertEqual(self.factory.opened, 1)
            self.assertEqual(self.factory.closed, 0)
            self.assertEqual(self.preview.status()["state"], "LIVE")

    def test_display_encoding_preserves_original_rgb_and_depth(self):
        """표시용 압축과 색상화가 이후 인식에 사용할 원본 배열을 바꾸지 않는다."""
        rgb = np.full((18, 32, 3), 100, np.uint8)
        depth = np.full((16, 24), .25, np.float32)
        depth[0, 0] = np.nan
        rgb_before, depth_before = rgb.copy(), depth.copy()
        images = preview_images(RGBDFrame(rgb, depth, 1, 1, time.monotonic()))
        self.assertEqual(set(images), {"color", "depth"})
        self.assertTrue(all(data.startswith(b"\xff\xd8") for data in images.values()))
        np.testing.assert_array_equal(rgb, rgb_before)
        np.testing.assert_array_equal(depth, depth_before)


if __name__ == "__main__":
    unittest.main()
