"""관측한 평면에 대한 그리퍼 간격과 작은 상승의 위치·방향 변화를 검증한다."""

import unittest

import numpy as np

from planning.beam_lift import BeamLiftPlanner, LiftSettings


class BeamLiftTests(unittest.TestCase):
    """실물에서 저장한 관절 자세를 하나의 참고값으로 사용해 여러 간격을 확인한다."""

    def setUp(self):
        """정면 보정 후 참고 자세와 열린 그리퍼 상태를 준비한다."""
        self.q = np.deg2rad([-11.6015625, -47.109375, 12.568359375, 62.05078125,
                            3.69140625, 78.837890625, 22.1484375])
        self.grippers = {"G_L": 4.21875, "G_R": -119.8828125}
        self.planner = BeamLiftPlanner()

    def test_gap_uses_gripper_top_instead_of_camera_distance(self):
        """카메라 깊이를 그리퍼 간격으로 오인하지 않고 CAD 상단까지의 길이를 제외한다."""
        for distance in (.22, .2457, .27):
            gap = self.planner.gap(self.q, self.grippers, [0, 0, -1], distance)
            self.assertAlmostEqual(gap, distance - .19145, delta=.00001)
        with self.assertRaises(ValueError):
            self.planner.gap(self.q, self.grippers, [0, 0, -1], -1.)
        with self.assertRaises(ValueError):
            LiftSettings(gap_m=.001)

    def test_small_lift_reduces_gap_and_keeps_optical_axis(self):
        r"""현재 정면을 유지한 작은 이동에서 새 평면 좌표로 계산한 간격이 계획량만큼 줄어든다.

        $$d_{next}=d+n_W^T(p_{C,next}-p_C)$$
        """
        fk = self.planner.solver.fk
        before = fk.depth_camera_pose(self.q)
        tip_before = fk.forward(self.q).T_world_tip_R.copy()
        for distance in (.222, .2457, .27):
            with self.subTest(distance=distance):
                normal = np.array([0., 0., -1.])
                step = self.planner.plan(self.q, self.grippers, normal, distance)
                self.assertEqual(step.kind, "lift")
                self.assertLessEqual(step.distance_m, .005)
                self.assertLessEqual(np.max(np.abs(np.rad2deg(step.q_rad - self.q))), 3.)
                after = fk.depth_camera_pose(step.q_rad)
                world_normal = before[:3, :3] @ normal
                next_normal = after[:3, :3].T @ world_normal
                # 카메라 원점 이동에 따른 같은 평면의 새 상수항: $$d_{next}=d+n_W^T(p_{C,next}-p_C)$$
                next_offset = distance + world_normal @ (after[:3, 3] - before[:3, 3])
                next_gap = self.planner.gap(step.q_rad, self.grippers, next_normal, next_offset)
                self.assertAlmostEqual(next_gap, step.gap_before_m - step.distance_m, delta=.0001)
                np.testing.assert_allclose(after[:3, 2], before[:3, 2], atol=1e-6)
                np.testing.assert_allclose(fk.forward(step.q_rad).T_world_tip_R[:3, :3], tip_before[:3, :3], atol=1e-6)

    def test_alignment_can_recover_tilt_without_rising(self):
        """상승 중 기울기가 생기면 높이 이동 전에 현재 팁 위치에서 정면을 보정한다."""
        normal = np.array([.04, 0., -1.])
        normal /= np.linalg.norm(normal)
        step = self.planner.plan(self.q, self.grippers, normal, .2457)
        self.assertEqual(step.kind, "alignment")
        self.assertEqual(step.distance_m, 0.)
        with self.assertRaises(ValueError):
            self.planner.plan(self.q, self.grippers, [0, 0, -1], .21645)


if __name__ == "__main__":
    unittest.main()
