"""관측 시작 자세가 높이만 낮추고 고정 팁·방향·카메라 정면을 유지하는지 검증한다."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kinematics.ik import IKSettings, InverseKinematics
from planning.observation_pose import plan_observation_pose
from simulation.beam import beam_reference
from simulation.model import load_model


class ObservationPoseTests(unittest.TestCase):
    """독립 MuJoCo 상태로 시작 자세의 기하 조건과 실패 처리를 확인한다."""

    def setUp(self) -> None:
        """기존 이십 센티미터 파지 탐색에서 얻은 기준각과 단일 시작값 IK를 준비한다."""
        self.q_grasp = np.array([
            -0.3721296318898312, -0.8338212980274553, -1.9879711554845417e-7,
            1.4739499730800456, -1.989686261166496e-7,
            0.8338213822611935, 0.37212936457085627,
        ])
        self.solver = InverseKinematics(settings=IKSettings(starts=1))

    def test_height_only_and_camera_front_view(self) -> None:
        r"""높이를 바꿔도 고정 팁·빔 배치·이동 팁 방향과 카메라 정면 조건이 유지된다.

        $$\Delta p_R=(0,0,-h)^T$$
        """
        original = self.q_grasp.copy()
        _, before = load_model(self.q_grasp)
        for drop_m in (.03, .05, .10):
            with self.subTest(drop_m=drop_m):
                pose = plan_observation_pose(self.q_grasp, drop_m=drop_m, solver=self.solver)
                model, after = load_model(pose.q_rad)
                # 전후 오른쪽 팁의 실제 변위: $$\Delta p_R=p_{R,after}-p_{R,before}$$
                displacement = after.site("tip_R").xpos - before.site("tip_R").xpos
                assert_allclose(displacement, [0, 0, -drop_m], atol=1e-6)
                assert_allclose(after.site("tip_R").xmat, before.site("tip_R").xmat, atol=1e-6)
                assert_allclose(after.site("tip_L").xpos, before.site("tip_L").xpos, atol=1e-12)
                assert_allclose(after.site("tip_L").xmat, before.site("tip_L").xmat, atol=1e-12)
                assert_array_equal(after.body("ibeam").xpos, before.body("ibeam").xpos)
                for name in ("G_L", "G_R"):
                    assert_array_equal(after.joint(name).qpos, before.joint(name).qpos)
                assert_allclose(beam_reference(model, after)["normal"], [0, 0, -1], atol=1e-6)
                assert_allclose(after.site("tip_R").xpos, pose.target_world[:3, 3], atol=1e-6)
                limits = self.solver.fk.joint_limits
                self.assertTrue(np.all(pose.q_rad >= limits[:, 0]))
                self.assertTrue(np.all(pose.q_rad <= limits[:, 1]))
        assert_array_equal(self.q_grasp, original)

    def test_invalid_drop_is_rejected(self) -> None:
        """양수가 아니거나 유한하지 않은 높이는 IK 실행 전에 거부한다."""
        for drop_m in (0, -.01, np.nan, np.inf):
            with self.subTest(drop_m=drop_m), self.assertRaises(ValueError):
                plan_observation_pose(self.q_grasp, drop_m=drop_m, solver=self.solver)

    def test_unreachable_pose_does_not_return_joint_command(self) -> None:
        """도달할 수 없는 높이에서는 사용할 관절각을 반환하지 않는다."""
        with self.assertRaisesRegex(ValueError, "관측 시작 자세를 구하지 못했습니다"):
            plan_observation_pose(self.q_grasp, drop_m=2., solver=self.solver)


if __name__ == "__main__":
    unittest.main()
