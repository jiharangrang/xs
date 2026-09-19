"""서로 다른 기울기와 관절 자세에서 보정 방향·이동 크기·팁 위치 보존을 검증한다."""

import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from kinematics.joints import ARM_JOINT_NAMES
from planning.beam_alignment import AlignmentSettings, BeamAlignmentPlanner, tilt_degrees
from simulation.model import load_model


class BeamAlignmentTests(unittest.TestCase):
    """계획기와 별도로 갱신한 MuJoCo 상태에서 보정의 기하학적 결과를 확인한다."""

    def setUp(self):
        """빔 아래 기준 자세와 공통 모델을 사용하는 계획기를 준비한다."""
        self.q = np.deg2rad([-19.95, -35.62, 2.06, 79.51, 1.33, 64.89, 22.19])
        self.planner = BeamAlignmentPlanner()

    def test_varied_tilt_signs_and_poses_reduce_error_with_fixed_tip(self):
        r"""기울기의 부호와 현재 자세가 달라도 독립 FK에서 오차가 줄고 팁은 유지된다.

        $$n_{next}=R_{next}^T R_{before}n$$
        """
        model, data = load_model(self.q)
        for offset in (0., .01):
            q = self.q.copy()
            q[0] += offset
            for xy in ((.12, -.15), (-.12, .15), (.15, .1), (-.1, -.15), (0., .15)):
                with self.subTest(offset=offset, xy=xy):
                    normal = np.array([*xy, -1.])
                    normal /= np.linalg.norm(normal)
                    before = self.planner.fk.depth_camera_pose(q)
                    tip_before = self.planner.fk.forward(q).T_world_tip_R[:3, 3]
                    step = self.planner.plan(q, normal, q)
                    for name, angle in zip(ARM_JOINT_NAMES, step.q_rad):
                        data.joint(name).qpos[0] = angle
                    mujoco.mj_forward(model, data)
                    rotation = data.camera("gemini215_depth").xmat.reshape(3, 3) @ np.diag([1, -1, -1])
                    # 독립 모델로 계산한 이동 후 관측 법선: $$n_{next}=R_{next}^T R_{before}n$$
                    actual_normal = rotation.T @ before[:3, :3] @ normal
                    self.assertLess(tilt_degrees(actual_normal), tilt_degrees(normal) - .05)
                    np.testing.assert_allclose(actual_normal, step.predicted_normal, atol=1e-9)
                    np.testing.assert_allclose(data.site("tip_R").xpos, tip_before, atol=1e-5)
                    self.assertLessEqual(np.max(np.abs(np.rad2deg(step.q_rad - q))), 3.00001)

    def test_repeated_reobservation_converges_for_different_initial_errors(self):
        r"""고정된 공간 평면을 매번 새 카메라에서 관측하면 여러 초기 오차가 정면으로 수렴한다.

        $$n_C=R_{WC}^T n_W$$
        """
        for vector in ((.07, .06, 0.), (-.12, -.07, 0.), (.03, -.09, 0.)):
            with self.subTest(vector=vector):
                q = self.q.copy()
                normal = Rotation.from_rotvec(vector).apply([0., 0., -1.])
                world_normal = self.planner.fk.depth_camera_pose(q)[:3, :3] @ normal
                for _ in range(20):
                    # 새 관절각에 대응하는 카메라의 실제 관측: $$n_C=R_{WC}^T n_W$$
                    normal = self.planner.fk.depth_camera_pose(q)[:3, :3].T @ world_normal
                    if tilt_degrees(normal) <= 1:
                        break
                    q = self.planner.plan(q, normal, self.q).q_rad
                self.assertLessEqual(tilt_degrees(normal), 1.)

    def test_invalid_normal_and_motion_envelope_are_rejected(self):
        """비정상 법선과 과도한 관절 보정은 실행할 목표를 만들지 않는다."""
        for normal in ([0, 0, 0], [0, 0, 1], [np.nan, 0, -1], [1, 0, -1]):
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                self.planner.plan(self.q, normal, self.q)
        planner = BeamAlignmentPlanner(settings=AlignmentSettings(max_total_joint_deg=.01))
        with self.assertRaises(ValueError):
            planner.plan(self.q, [.12, -.15, -1], self.q)
        result = self.planner.plan(self.q, [0, 0, -1], self.q)
        np.testing.assert_array_equal(result.q_rad, self.q)


if __name__ == "__main__":
    unittest.main()
