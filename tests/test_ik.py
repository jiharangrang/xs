"""목표 자세 도달, 관절 제한의 출처, 비용과 실패 시 반환 계약을 검증한다."""

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import check_grad
import yaml

from hardware.joint_control import load_joint_calibration
from kinematics.fk import DEFAULT_MODEL_PATH, ForwardKinematics
from kinematics.ik import IKCandidate, IKSettings, InverseKinematics, joint_motion_cost, joint_motion_gradient
from kinematics.joints import ARM_JOINT_NAMES
from kinematics.joint_limits import DEFAULT_CALIBRATION_PATH
from kinematics.poses import as_pose, pose_error
from planning.targets import beam_grasp_target


class InverseKinematicsTests(unittest.TestCase):
    """풀이기의 상태 메시지와 별개로 FK 및 MuJoCo 좌표를 확인한다."""

    def test_beam_goal_matches_independent_mujoco_state(self) -> None:
        """빔 앞쪽 목표의 해를 별도 MuJoCo 상태에 넣어 위치·방향을 검증한다."""
        target = beam_grasp_target()
        q_start = np.zeros(7)
        result = InverseKinematics().solve(target, q_start)
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.solved_attempts, 0)
        self.assertEqual(result.attempts, 24)
        self.assertGreater(len(result.candidates), 0)
        self.assertLessEqual(len(result.candidates), 5)
        self.assertEqual(result.cost, result.candidates[0].cost)
        assert_array_equal(result.q_rad, result.candidates[0].q_rad)
        assert_array_equal(q_start, np.zeros(7))
        assert_allclose(target[:3, 3], [-0.1, 0, 0])
        assert_allclose(target[:3, :3], np.diag([-1, -1, 1]))

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        data = mujoco.MjData(model)
        for candidate in result.candidates:
            for name, angle in zip(ARM_JOINT_NAMES, candidate.q_rad, strict=True):
                joint = model.joint(name)
                self.assertGreaterEqual(angle, joint.range[0])
                self.assertLessEqual(angle, joint.range[1])
                data.joint(name).qpos[0] = angle
            mujoco.mj_forward(model, data)
            assert_allclose(data.site("tip_R").xpos, [0.1, -0.03636, 0.241530662041], atol=1e-5)
            assert_allclose(data.site("tip_R").xmat.reshape(3, 3), np.eye(3), atol=1e-5)
            assert_allclose(data.site("tip_L").xpos, [0, -0.03636, 0.241530662041], atol=1e-12)

    def test_candidates_are_sorted_deduplicated_and_bounded(self) -> None:
        """비슷한 해는 더 싼 것으로 교체하고 서로 다른 상위 후보만 제한해 보관한다."""
        solver = InverseKinematics(settings=IKSettings(max_candidates=2))
        candidates = []
        for angle in (0.8, 0.5005, -0.8, 0.5, 1.0):
            q = np.array([angle, 0, 0, 0, 0, 0, 0])
            candidate = IKCandidate(q, joint_motion_cost(q, np.zeros(7)), 0.0, 0.0)
            candidates = solver._retain_candidate(candidates, candidate)
            self.assertLessEqual(len(candidates), 2)
        assert_allclose([item.q_rad[0] for item in candidates], [0.5, 0.8])
        self.assertLess(candidates[0].cost, candidates[1].cost)

    def test_already_at_target_needs_no_optimization(self) -> None:
        """특이한 초기 자세라도 이미 목표라면 비용 영의 해를 바로 반환한다."""
        q_start = np.zeros(7)
        target = ForwardKinematics().forward(q_start).T_tip_L_tip_R
        result = InverseKinematics().solve(target, q_start)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(result.cost, 0.0)
        self.assertEqual(len(result.candidates), 1)
        assert_array_equal(result.q_rad, q_start)
        result.q_rad[0] = 1
        assert_array_equal(q_start, np.zeros(7))

    def test_cost_gradient_and_equal_joint_weights(self) -> None:
        """비용 미분을 수치 미분과 비교하고 관절 순서에 따른 선호가 없는지 확인한다."""
        q_start = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
        q_candidate = np.array([-0.3, 0.5, 0.7, -0.9, 0.8, 0.1, -0.2])
        gradient_error = check_grad(joint_motion_cost, joint_motion_gradient, q_candidate, q_start)
        self.assertLess(gradient_error, 1e-6)
        self.assertAlmostEqual(
            joint_motion_cost(q_candidate, q_start),
            joint_motion_cost(q_candidate[::-1], q_start[::-1]),
        )

    def test_failed_search_does_not_return_command_angles(self) -> None:
        """제한된 탐색에서 해를 찾지 못하면 사용할 관절각을 반환하지 않는다."""
        solver = InverseKinematics(settings=IKSettings(starts=2, max_iterations=40))
        result = solver.solve(beam_grasp_target(10.0), np.zeros(7))
        self.assertFalse(result.success)
        self.assertIsNone(result.q_rad)
        self.assertIsNone(result.cost)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.candidates, ())

    def test_limits_follow_calibration_for_both_ik_and_hardware(self) -> None:
        """캘리브레이션 변경이 XML 수정 없이 IK와 실물 제어에 같은 범위로 적용된다."""
        document = yaml.safe_load(DEFAULT_CALIBRATION_PATH.read_text())
        document["joints"]["J1"].update(lower_rad=-0.1, upper_rad=0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            path.write_text(yaml.safe_dump(document))
            solver = InverseKinematics(calibration_path=path)
            assert_allclose(solver.fk.joint_limits[0], [-0.1, 0.2])
            _, hardware = load_joint_calibration(path)
            hardware_limits = [[hardware[name].lower_deg, hardware[name].upper_deg] for name in ARM_JOINT_NAMES]
            assert_allclose(np.rad2deg(solver.fk.joint_limits), hardware_limits)
            with self.assertRaises(ValueError):
                solver.solve(beam_grasp_target(), [0.3, 0, 0, 0, 0, 0, 0])

    def test_new_one_sided_limits_reject_previously_allowed_start_poses(self) -> None:
        """각기 제한된 세 방향의 범위 밖 시작 자세를 이미 목표인 경우에도 거부한다."""
        solver = InverseKinematics()
        for index, angle in ((1, 30), (3, -35), (5, -20)):
            q_start = np.zeros(7)
            q_start[index] = np.deg2rad(angle)
            target = solver.fk.forward(q_start).T_tip_L_tip_R
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "캘리브레이션"):
                solver.solve(target, q_start)

    def test_rejects_invalid_pose_angles_and_settings(self) -> None:
        """잘못된 입력을 계산 시작 전에 거부하는지 확인한다."""
        solver = InverseKinematics()
        for pose in (np.eye(3), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                solver.solve(pose, np.zeros(7))
        for angles in ([0] * 6, [np.inf] * 7, [4.0] * 7):
            with self.subTest(angles=angles), self.assertRaises(ValueError):
                solver.solve(beam_grasp_target(), angles)
        for settings in (
            {"starts": 0}, {"max_iterations": -1}, {"random_seed": -1},
            {"position_scale_m": 0}, {"max_candidates": 0}, {"duplicate_tolerance_rad": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                IKSettings(**settings)

    def test_half_turn_is_not_mistaken_for_zero_rotation_error(self) -> None:
        """반 바퀴 다른 방향이 영의 회전 오차로 잘못 계산되지 않는지 확인한다."""
        actual = as_pose(np.diag([-1.0, -1.0, 1.0, 1.0]))
        error = pose_error(actual, np.eye(4))
        assert_allclose(error[:3], np.zeros(3))
        self.assertAlmostEqual(np.linalg.norm(error[3:]), np.pi)


if __name__ == "__main__":
    unittest.main()
