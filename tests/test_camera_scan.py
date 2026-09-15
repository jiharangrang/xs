"""실물 없이 스캔 원본과 촬영 시점의 자세·모터 로그 저장을 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hardware.camera import CameraError, RGBDFrame
from hardware.camera_scan import capture_scan


class CameraScanTests(unittest.TestCase):
    """프레임 수와 저장값, 실패 시 자원 반환을 확인한다."""

    def test_burst_preserves_frames_pose_timing_and_matching_log(self) -> None:
        """준비 프레임을 제외한 원본 15개와 실제 자세·같은 구간의 로그를 저장한다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "motors.jsonl"
            log.write_text('{"event": "settings"}\n')
            with patch("hardware.camera_scan.Gemini215Camera") as factory:
                camera = factory.return_value.__enter__.return_value
                camera.info = {"aligned": False, "depth_unit": "m"}
                sequence = []

                def read() -> RGBDFrame:
                    """프레임마다 다른 원본 값과 모터 기록을 준비한다."""
                    index = len(sequence) + 1
                    sequence.append(index)
                    with log.open("a") as stream:
                        stream.write(json.dumps({"event": "sample", "index": index}) + "\n")
                    depth = np.full((2, 3), index / 100, np.float32)
                    depth[0, 0] = np.nan
                    return RGBDFrame(np.full((3, 4, 3), index, np.uint8), depth,
                                     index * 66000, index * 66000 + 100, index / 15)

                camera.read.side_effect = read
                poses = []

                def read_pose() -> dict:
                    """촬영 전후에 측정한 실제 관절각을 반환한다."""
                    pose = {"angles_deg": {"J1": 12.3}, "states": [{"name": "J1", "speed_deg_s": 0}],
                            "received_monotonic_s": len(sequence) / 15}
                    poses.append(pose)
                    return pose

                result = capture_scan(read_pose, log, root / "scans")
                output = Path(result["path"])
                metadata = json.loads((output / "capture.json").read_text())
                self.assertEqual(camera.read.call_count, 45)
                self.assertEqual(result["frames"], 15)
                self.assertEqual(metadata["joints_before"], poses[0])
                self.assertEqual(metadata["joints_after"], poses[1])
                self.assertEqual(metadata["samples"][0]["color_timestamp_us"], 31 * 66000)
                with np.load(output / "depth_frames.npz") as depths:
                    self.assertEqual(len(depths.files), 15)
                    self.assertAlmostEqual(float(depths["frame_000"][1, 1]), .31)
                    self.assertTrue(np.isnan(depths["frame_000"][0, 0]))
                    np.testing.assert_array_equal(depths["frame_014"], np.load(output / "depth_m.npy"))
                self.assertEqual(len(list((output / "frames").glob("*_rgb.png"))), 15)
                self.assertEqual((output / "rgb.png").read_bytes(), (output / "frames/014_rgb.png").read_bytes())
                rows = [json.loads(line) for line in (output / "motors.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 45)
                self.assertEqual(metadata["motor_log_source"], str(log))
                factory.return_value.__exit__.assert_called_once()

    def test_camera_closes_on_capture_error_without_reporting_saved(self) -> None:
        """영상 수신 실패 시에도 카메라를 닫고 저장 완료로 보고하지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "motors.jsonl"
            log.touch()
            with patch("hardware.camera_scan.Gemini215Camera") as factory:
                factory.return_value.__enter__.return_value.read.side_effect = CameraError("수신 실패")
                with self.assertRaisesRegex(CameraError, "수신 실패"):
                    capture_scan(lambda: {}, log, root / "scans")
                factory.return_value.__exit__.assert_called_once()
                self.assertFalse((root / "scans").exists())
