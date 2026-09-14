"""카메라 원본 영상의 단위·색상·버퍼 수명과 연결 오류 처리를 검증한다."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hardware.camera import CameraError, Gemini215Camera, _decode_depth, _decode_rgb


def frame_buffer(data, image_format, *, scale=1.0):
    """SDK 프레임과 같은 읽기 인터페이스로 작은 시험 버퍼를 제공한다."""
    return SimpleNamespace(
        get_data=lambda: data,
        get_format=lambda: SimpleNamespace(name=image_format),
        get_width=lambda: data.shape[1],
        get_height=lambda: data.shape[0],
        get_depth_scale=lambda: scale,
    )


class CameraInputTests(unittest.TestCase):
    """실물 없이 원본 값의 보존과 오류 경계를 확인한다."""

    def test_depth_uses_device_scale_and_marks_invalid(self):
        """깊이 배율이 다른 장치에서도 미터 단위와 무효 픽셀을 구분한다."""
        raw = np.array([[0, 1000, 2000]], dtype=np.uint16)
        depth = _decode_depth(frame_buffer(raw, "Y16", scale=0.25))
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isnan(depth[0, 0]))
        np.testing.assert_allclose(depth[0, 1:], [0.25, 0.5])
        raw[:] = 123
        np.testing.assert_allclose(depth[0, 1:], [0.25, 0.5])

    def test_depth_rejects_invalid_scale_and_format(self):
        """잘못된 깊이 배율이나 형식을 실제 거리로 반환하지 않는다."""
        raw = np.ones((2, 2), dtype=np.uint16)
        for scale in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(CameraError):
                _decode_depth(frame_buffer(raw, "Y16", scale=scale))
        with self.assertRaises(CameraError):
            _decode_depth(frame_buffer(raw, "Y8"))

    def test_color_order_and_buffer_independence(self):
        """RGB·BGR 입력을 RGB로 통일하고 SDK 버퍼 재사용의 영향을 없앤다."""
        try:
            import cv2
        except ImportError:
            self.skipTest("camera 선택 의존성이 필요합니다.")
        source = np.array([[[255, 0, 10]]], dtype=np.uint8)
        rgb = _decode_rgb(frame_buffer(source, "RGB"))
        bgr = _decode_rgb(frame_buffer(source, "BGR"))
        source[:] = 0
        np.testing.assert_array_equal(rgb, [[[255, 0, 10]]])
        np.testing.assert_array_equal(bgr, [[[10, 0, 255]]])

    def test_mjpeg_decode_and_corruption(self):
        """JPEG 색상 순서와 손상된 데이터의 거부를 확인한다."""
        try:
            import cv2
        except ImportError:
            self.skipTest("camera 선택 의존성이 필요합니다.")
        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        _, encoded = cv2.imencode(".jpg", bgr)
        frame = frame_buffer(bgr, "MJPG")
        frame.get_data = lambda: encoded
        rgb = _decode_rgb(frame)
        self.assertGreater(rgb[0, 0, 0], 250)
        self.assertLess(rgb[0, 0, 2], 5)
        frame.get_data = lambda: b"not a jpeg"
        with self.assertRaises(CameraError):
            _decode_rgb(frame)

    def test_read_requires_open_and_finite_wait(self):
        """미연결 상태나 무효 대기 시간을 명확하게 거부한다."""
        camera = Gemini215Camera()
        with self.assertRaises(CameraError):
            camera.read()
        camera._pipeline = MagicMock()
        for timeout in (0, -1, True, float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                camera.read(timeout_ms=timeout)

    def test_incomplete_frames_do_not_return_stale_images(self):
        """RGB나 깊이가 빠진 프레임을 이전 영상과 섞지 않고 제한 시간에 실패한다."""
        camera = Gemini215Camera()
        incomplete = SimpleNamespace(get_color_frame=lambda: object(), get_depth_frame=lambda: None)
        camera._pipeline = MagicMock()
        camera._pipeline.wait_for_frames.return_value = incomplete
        with patch("hardware.camera.time.monotonic", side_effect=[0, 0, 0, 1]):
            with self.assertRaisesRegex(CameraError, "받지 못했습니다"):
                camera.read(timeout_ms=100)
        camera._pipeline.wait_for_frames.assert_called_once_with(100)

    def test_sdk_read_error_is_reported(self):
        """장치 분리 등 SDK 수신 오류를 호출자에게 전달한다."""
        camera = Gemini215Camera()
        camera._pipeline = MagicMock()
        camera._pipeline.wait_for_frames.side_effect = RuntimeError("disconnected")
        with self.assertRaisesRegex(CameraError, "disconnected"):
            camera.read()

    def test_failed_start_releases_resources(self):
        """스트림 시작 실패 뒤 장치를 해제해 다음 연결 시도를 방해하지 않는다."""
        sdk = MagicMock()
        devices = sdk.Context.return_value.query_devices.return_value
        devices.get_count.return_value = 1
        devices.get_device_name_by_index.return_value = "Orbbec Gemini 215"
        pipeline = sdk.Pipeline.return_value
        pipeline.start.side_effect = RuntimeError("start failed")
        camera = Gemini215Camera()
        with patch("hardware.camera._load_sdk", return_value=sdk):
            with self.assertRaisesRegex(CameraError, "start failed"):
                camera.open()
        pipeline.stop.assert_called_once()
        self.assertIsNone(camera._pipeline)
        self.assertIsNone(camera._context)

    def test_multiple_cameras_require_selection(self):
        """여러 카메라가 연결되면 임의의 카메라를 열지 않는다."""
        sdk = MagicMock()
        devices = sdk.Context.return_value.query_devices.return_value
        devices.get_count.return_value = 2
        devices.get_device_name_by_index.return_value = "Orbbec Gemini 215"
        with patch("hardware.camera._load_sdk", return_value=sdk):
            with self.assertRaisesRegex(CameraError, "2개"):
                Gemini215Camera().open()
        devices.get_device_by_index.assert_not_called()

    def test_close_is_idempotent(self):
        """중복 종료 시 스트림을 한 번만 중단한다."""
        camera = Gemini215Camera()
        pipeline = camera._pipeline = MagicMock()
        camera.close()
        camera.close()
        pipeline.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
