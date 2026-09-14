"""카메라 투영·깊이 단위와 실측 입력의 비교 조건을 검증한다."""

import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np

from scripts.compare_camera import read_joint_snapshot
from simulation.camera import (
    JOINT_NAMES, _configure_color_pose, _configure_projection,
    reference_camera_info, render_camera,
)
from kinematics.fk import DEFAULT_MODEL_PATH
from perception.depth import rectify_image


class SimulationCameraTests(unittest.TestCase):
    """장치 없이 알려진 형상과 렌즈 설정으로 비교 통로를 확인한다."""

    def test_projection_and_optical_depth_against_known_box(self):
        """주점이 중심에서 벗어난 카메라로 상자의 투영 위치와 수직 깊이를 확인한다."""
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody>
              <camera name="test" pos="0 0 0" quat="0 1 0 0"
                resolution="160 120" sensorsize="1 1" focalpixel="100 100"/>
              <geom type="box" pos="0 0 .4" size=".1 .07 .001"/>
            </worldbody></mujoco>''')
        profile = {"width": 160, "height": 120,
                   "intrinsics": {"fx": 170, "fy": 180, "cx": 68, "cy": 67}}
        _configure_projection(model, 0, profile)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            renderer.update_scene(data, camera="test")
            renderer.enable_depth_rendering()
            depth = renderer.render()
        rows, columns = np.where(depth < 0.5)
        self.assertAlmostEqual(float(columns.mean()), 68, delta=1)
        self.assertAlmostEqual(float(rows.mean()), 67, delta=1)
        self.assertAlmostEqual(float(np.ptp(columns)), 85, delta=2)
        self.assertAlmostEqual(float(np.ptp(rows)), 63, delta=2)
        np.testing.assert_allclose(depth[rows, columns], 0.399, atol=1e-5)

    def test_factory_transform_direction_and_units(self):
        """깊이에서 RGB로 주어진 이동량의 역변환과 밀리미터 단위를 확인한다."""
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        _configure_color_pose(model, {"rotation": np.eye(3).tolist(), "translation_mm": [-20, 1, 2]})
        np.testing.assert_allclose(model.camera("gemini215_rgb").pos - model.camera("gemini215_depth").pos,
                                   [0.020, -0.001, -0.002], atol=1e-10)
        np.testing.assert_allclose(model.camera("gemini215_rgb").quat,
                                   model.camera("gemini215_depth").quat, atol=1e-10)

    def test_incomplete_or_nonfinite_pose_is_rejected(self):
        """누락된 관절을 몰래 영점으로 채우거나 무효 각도를 렌더링하지 않는다."""
        with self.assertRaises(ValueError):
            render_camera({"J1": 0.0})
        pose = dict.fromkeys(JOINT_NAMES, 0.0)
        pose["J7"] = float("nan")
        with self.assertRaises(ValueError):
            render_camera(pose)

    def test_depth_rectification_preserves_invalid_pixels(self):
        """왜곡 없는 원본 깊이의 거리와 NaN을 그대로 보존한다."""
        profile = reference_camera_info()["depth"]
        profile.update(width=2, height=2)
        depth = np.array([[0.17, np.nan], [0.22, 0.31]], dtype=np.float32)
        result = rectify_image(depth, profile, depth=True)
        np.testing.assert_array_equal(result, depth)
        self.assertFalse(np.shares_memory(result, depth))
        with self.assertRaises(ValueError):
            rectify_image(depth[:1], profile, depth=True)

    def test_motor_read_error_is_not_treated_as_zero(self):
        """통신 오류를 원점으로 간주해 잘못된 비교 결과를 만들지 않는다."""
        socket = MagicMock()
        socket.__enter__.return_value.recv.return_value = '{"joints": [], "error": "disconnected"}'
        with patch("websockets.sync.client.connect", return_value=socket):
            with self.assertRaises(ValueError):
                read_joint_snapshot("ws://127.0.0.1:8000/ws")


if __name__ == "__main__":
    unittest.main()
