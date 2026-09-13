"""숫자로 지정한 관절각이 뷰어용 MuJoCo 상태에 그대로 반영되는지 검증한다."""

from contextlib import nullcontext
import unittest
from unittest.mock import MagicMock, patch

import glfw
import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kinematics.fk import ForwardKinematics
from kinematics.joints import ARM_JOINT_NAMES
from simulation.model import load_model, set_arm_angles
from simulation.viewer import show_pose


class PoseViewerModelTests(unittest.TestCase):
    """GUI 창을 열지 않고 입력 각도와 양쪽 팁 좌표의 일치를 확인한다."""

    def test_supplied_angles_are_not_clipped_and_match_fk(self) -> None:
        """시각화 입력을 임의로 제한하지 않고 FK와 동일한 자세를 표시한다."""
        q_rad = np.array([3.1, -0.4, 0.3, 0.6, -0.5, 0.1, 0.25])
        saved = q_rad.copy()
        model, data = load_model(q_rad)
        for name, angle in zip(ARM_JOINT_NAMES, q_rad, strict=True):
            self.assertEqual(data.joint(name).qpos[0], angle)
        result = ForwardKinematics().forward(q_rad)
        for name, transform in (("tip_L", result.T_world_tip_L), ("tip_R", result.T_world_tip_R)):
            assert_allclose(data.site(name).xpos, transform[:3, 3], atol=1e-12)
            assert_allclose(data.site(name).xmat.reshape(3, 3), transform[:3, :3], atol=1e-12)
        assert_array_equal(q_rad, saved)

    def test_change_angles_preserves_beam_and_gripper_state(self) -> None:
        """팔 자세를 갱신해도 빔과 손가락 관절 상태가 유지되는지 확인한다."""
        model, data = load_model()
        beam_position = data.geom("ibeam_mesh").xpos.copy()
        data.joint("G_L").qpos[0] = 0.3
        data.joint("G_R").qpos[0] = -0.2
        set_arm_angles(model, data, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert_array_equal(data.geom("ibeam_mesh").xpos, beam_position)
        self.assertEqual(data.joint("G_L").qpos[0], 0.3)
        self.assertEqual(data.joint("G_R").qpos[0], -0.2)

    def test_candidate_key_cycles_and_restores_selected_pose_only_when_enabled(self) -> None:
        """후보 호출에서만 Enter가 작동하고 순환·초기 자세 비교·선택 후보 복원이 이어지는지 확인한다."""
        first = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        second = -first
        for candidate_mode in (False, True):
            with self.subTest(candidate_mode=candidate_mode):
                model, data = load_model(first)
                viewer = MagicMock()
                viewer.__enter__.return_value = viewer
                viewer.lock.side_effect = nullcontext
                viewer.cam = mujoco.MjvCamera()
                viewer.opt = mujoco.MjvOption()
                keys = iter((None, glfw.KEY_ENTER, ord(" "), ord("R"), glfw.KEY_ENTER))
                snapshots = []
                with (
                    patch("simulation.viewer.load_model", return_value=(model, data)),
                    patch("simulation.viewer.mujoco.viewer.launch_passive", return_value=viewer) as launch,
                    patch("simulation.viewer.time.sleep"),
                    patch("builtins.print"),
                ):
                    def is_running() -> bool:
                        """각 화면 갱신 전에 미리 정한 키를 한 번씩 전달한다."""
                        try:
                            key = next(keys)
                        except StopIteration:
                            return False
                        if key is not None:
                            launch.call_args.kwargs["key_callback"](key)
                        return True

                    def record_pose() -> None:
                        """화면으로 전달될 일곱 관절각을 복사한다."""
                        snapshots.append([data.joint(name).qpos[0] for name in ARM_JOINT_NAMES])

                    viewer.is_running.side_effect = is_running
                    viewer.sync.side_effect = record_pose
                    show_pose(first, other_candidates=[second] if candidate_mode else None)
                selected = second if candidate_mode else first
                assert_allclose(snapshots, [first, selected, np.zeros(7), selected, first])
                overlay = viewer.set_texts.call_args.args[0]
                self.assertEqual("Enter: next candidate" in overlay[2], candidate_mode)
                if candidate_mode:
                    self.assertIn("Candidate 1/2", overlay[3])


if __name__ == "__main__":
    unittest.main()
