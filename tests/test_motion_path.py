"""경로 저장·복원 후 같은 재생 상태가 나오고 고정단 전환 때 배치가 이어지는지 확인한다."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import yaml

from kinematics.anchoring import TipAnchor
from kinematics.fk import ForwardKinematics
from kinematics.joints import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
from kinematics.joint_limits import DEFAULT_CALIBRATION_PATH
from planning.motion_path import MotionPath, MotionSegment, load_path, save_path
from simulation.model import load_model, set_path_time
from simulation.viewer import show_path


class MotionPathTests(unittest.TestCase):
    """작은 두 구간 경로 하나로 저장 형식과 재생 좌표 변환을 함께 확인한다."""

    def test_roundtrip_interpolation_and_support_switch(self) -> None:
        """R 고정에서 L 고정으로 이어지는 경로를 저장·복원하고 실제 표시 상태를 FK와 비교한다."""
        q_before = np.array([0.2, -0.4, 0.3, 0.6, -0.5, 0.1, 0.25])
        q_after = np.array([0.22, -0.42, 0.31, 0.63, -0.52, 0.12, 0.23])
        fk = ForwardKinematics()
        right_anchor = TipAnchor("tip_R", fk.forward(q_before).T_world_tip_R)
        switch_pose = right_anchor.place(fk.forward(q_after))
        left_anchor = TipAnchor("tip_L", switch_pose.T_world_tip_L)
        original = MotionPath((
            MotionSegment("right_fixed", right_anchor, [0, 1], [q_before, q_after], [[-0.1, -0.2], [-0.3, -0.4]]),
            MotionSegment("left_fixed", left_anchor, [0, 1], [q_after, q_before], [[-0.3, -0.4], [-0.1, -0.2]]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            destination = save_path(original, Path(directory) / "path.json")
            restored = load_path(destination)
        self.assertEqual(restored.duration_s, 2.0)
        for before, after in zip(original.segments, restored.segments, strict=True):
            assert_array_equal(before.q_rad, after.q_rad)
            assert_array_equal(before.time_s, after.time_s)
            assert_array_equal(before.gripper_q_rad, after.gripper_q_rad)
            assert_array_equal(before.anchor.T_world_fixed_tip, after.anchor.T_world_fixed_tip)
        assert_allclose(restored.sample(0.5)[1], [0.21, -0.41, 0.305, 0.615, -0.51, 0.11, 0.24])
        assert_allclose(restored.sample(0.5)[2], [-0.2, -0.3])

        model, data = load_model(q_before)
        beam_position = data.body("ibeam").xpos.copy()
        for elapsed_s in (0, 0.5, 1, 1.000001, 1.5, 2, 0):
            with self.subTest(elapsed_s=elapsed_s):
                segment_index, q_rad, gripper_q_rad = restored.sample(elapsed_s)
                self.assertEqual(set_path_time(model, data, restored, elapsed_s), segment_index)
                expected = restored.segments[segment_index].anchor.place(fk.forward(q_rad))
                for name, pose in (("tip_L", expected.T_world_tip_L), ("tip_R", expected.T_world_tip_R)):
                    assert_allclose(data.site(name).xpos, pose[:3, 3], atol=1e-10)
                    assert_allclose(data.site(name).xmat.reshape(3, 3), pose[:3, :3], atol=1e-10)
                assert_allclose([data.joint(name).qpos[0] for name in ARM_JOINT_NAMES], q_rad)
                assert_allclose([data.joint(name).qpos[0] for name in GRIPPER_JOINT_NAMES], gripper_q_rad)
                assert_array_equal(data.body("ibeam").xpos, beam_position)

    def test_saved_path_is_rechecked_after_calibration_change(self) -> None:
        """저장할 때 유효했던 경로라도 제한을 줄인 뒤에는 재생용 로드를 거부한다."""
        angles = np.zeros((2, 7))
        angles[1, 1] = 0.2
        motion = MotionPath((MotionSegment("saved", TipAnchor("tip_R", np.eye(4)), [0, 1], angles),))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.yaml"
            calibration.write_bytes(DEFAULT_CALIBRATION_PATH.read_bytes())
            destination = save_path(motion, root / "path.json", calibration_path=calibration)
            document = yaml.safe_load(calibration.read_text())
            document["joints"]["J2"]["upper_rad"] = 0.1
            calibration.write_text(yaml.safe_dump(document))
            with self.assertRaisesRegex(ValueError, "J2"):
                load_path(destination, calibration_path=calibration)

    def test_outside_waypoints_and_grippers_cannot_be_saved_or_replayed(self) -> None:
        """제한 밖 중간 지점과 손가락 각도를 거부하며 기존 경로 파일을 보존한다."""
        motion = MotionPath((MotionSegment("check", TipAnchor("tip_R", np.eye(4)), [0, 1, 2], np.zeros((3, 7))),))
        with tempfile.TemporaryDirectory() as directory:
            destination = save_path(motion, Path(directory) / "path.json")
            saved = destination.read_bytes()
            for index, angle in ((1, 30), (3, -35), (5, -20)):
                motion.segments[0].q_rad[1, index] = np.deg2rad(angle)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    save_path(motion, destination)
                self.assertEqual(destination.read_bytes(), saved)
                with patch("simulation.viewer._show_viewer") as viewer, self.assertRaises(ValueError):
                    show_path(motion)
                viewer.assert_not_called()
                motion.segments[0].q_rad[1, index] = 0
            payload = json.loads(saved)
            payload["segments"][0]["gripper_q_rad"] = [[0, 0], [0.3, 0], [0, 0]]
            destination.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "G_L"):
                load_path(destination)


if __name__ == "__main__":
    unittest.main()
