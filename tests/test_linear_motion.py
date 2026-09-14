"""직선 경로의 고정단 유지와 이동 팁 오차를 검증하고 불완전한 경로의 반환을 막는다."""

import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from kinematics.anchoring import TipAnchor
from kinematics.fk import DEFAULT_MODEL_PATH
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import ARM_JOINT_NAMES
from planning.linear_motion import plan_linear_motion


class LinearMotionTests(unittest.TestCase):
    """기존 파지 자세에서 시작하는 직선 이동과 실패 계약을 검사한다."""

    def setUp(self) -> None:
        """팁 간격이 이십 센티미터인 파지 IK에서 얻은 관절각과 단일 시작값 IK를 준비한다."""
        self.q_start = np.array([
            -0.37212941711515096, -2.3077712878071632, -1.20734399978031e-7,
            -1.4739499731590264, -1.2073433746511056e-7,
            2.307771338941721, 0.37212957939444863,
        ])
        self.solver = InverseKinematics(settings=IKSettings(starts=1))
        self.start = self.solver.fk.forward(self.q_start)
        self.anchor = TipAnchor("tip_R", self.start.T_world_tip_R)

    def test_rear_pull_matches_independent_mujoco_between_waypoints(self) -> None:
        r"""독립적인 MuJoCo 배치와 더 촘촘한 표본으로 고정 R 및 직선 이동 L을 확인한다.

        $$
        {}^W T_B={}^W T_F({}^M T_F)^{-1}{}^M T_B
        $$

        F는 고정 팁, B는 로봇 뿌리, M은 XML 배치이다.
        계획기의 TipAnchor.place를 호출하지 않고 MuJoCo 상태에서 직접 배치한다.
        """
        saved = self.q_start.copy()
        displacement = np.array([0.10, 0.0, 0.0])
        result = plan_linear_motion(self.q_start, self.anchor, displacement, solver=self.solver)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.q_path_rad.shape, (21, 7))
        self.assertEqual(result.target_poses_world.shape, (21, 4, 4))
        assert_array_equal(self.q_start, saved)
        assert_array_equal(result.q_path_rad[0], saved)
        limits = self.solver.fk.joint_limits
        self.assertTrue(np.all(result.q_path_rad >= limits[:, 0]))
        self.assertTrue(np.all(result.q_path_rad <= limits[:, 1]))

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        data = mujoco.MjData(model)
        root = model.body("gripper_L")
        root_position = root.pos.copy()
        root_quaternion = root.quat.copy()
        root_rotation = np.empty(9)
        mujoco.mju_quat2Mat(root_rotation, root_quaternion)
        root_pose = np.eye(4)
        root_pose[:3, :3] = root_rotation.reshape(3, 3)
        root_pose[:3, 3] = root_position
        mujoco.mj_kinematics(model, data)
        beam_position = data.body("ibeam").xpos.copy()
        maximum_error = 0.0

        for index in range(20):
            for fraction in np.linspace(0, 1, 11):
                # 독립적인 검사 지점의 관절각 보간: $$q(u)=(1-u)q_k+u q_{k+1}$$
                q_sample = (1 - fraction) * result.q_path_rad[index] + fraction * result.q_path_rad[index + 1]
                root.pos[:] = root_position
                root.quat[:] = root_quaternion
                for name, angle in zip(ARM_JOINT_NAMES, q_sample, strict=True):
                    data.joint(name).qpos[0] = angle
                mujoco.mj_kinematics(model, data)
                fixed_pose = np.eye(4)
                fixed_pose[:3, :3] = data.site("tip_R").xmat.reshape(3, 3)
                fixed_pose[:3, 3] = data.site("tip_R").xpos
                # 고정 팁의 직접 측정값으로 뿌리 배치 계산: $${}^W T_B={}^W T_F({}^M T_F)^{-1}{}^M T_B$$
                placed_root = self.anchor.T_world_fixed_tip @ np.linalg.inv(fixed_pose) @ root_pose
                root.pos[:] = placed_root[:3, 3]
                mujoco.mju_mat2Quat(root.quat, placed_root[:3, :3].reshape(-1))
                mujoco.mj_kinematics(model, data)
                # 전체 직선에서 검사 지점의 진행 비율: $$s=(k+u)/K$$
                progress = (index + fraction) / 20
                # 월드에서 요구하는 L팁 위치: $$p_d=p_0+s\Delta p_W$$
                expected_position = self.start.T_world_tip_L[:3, 3] + progress * displacement
                # 독립적으로 측정한 위치 오차: $$e_p=\|p-p_d\|_2$$
                error = np.linalg.norm(data.site("tip_L").xpos - expected_position)
                maximum_error = max(maximum_error, error)
                self.assertLessEqual(error, self.solver.settings.position_tolerance_m)
                assert_allclose(data.site("tip_L").xmat.reshape(3, 3), self.start.T_world_tip_L[:3, :3], atol=1e-6)
                assert_allclose(data.site("tip_R").xpos, self.anchor.T_world_fixed_tip[:3, 3], atol=1e-12)
                assert_allclose(data.site("tip_R").xmat.reshape(3, 3), self.anchor.T_world_fixed_tip[:3, :3], atol=1e-12)
                assert_array_equal(data.body("ibeam").xpos, beam_position)
        self.assertLessEqual(maximum_error, 1e-4)

    def test_left_anchor_uses_world_displacement_after_rotation_and_translation(self) -> None:
        r"""월드 배치를 회전·이동해도 L 고정 및 R의 월드 변위 해석이 유지되는지 확인한다.

        $$
        \Delta p_W=R_{WM}\Delta p_M
        $$

        M은 원래 배치, W는 시험용으로 회전·이동한 월드이다.
        """
        world_shift = np.eye(4)
        world_shift[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.7]).as_matrix()
        world_shift[:3, 3] = [0.4, -0.2, 0.8]
        # 원래 L팁의 월드 배치 변경: $${}^W T_L={}^W T_M{}^M T_L$$
        fixed_pose = world_shift @ self.start.T_world_tip_L
        anchor = TipAnchor("tip_L", fixed_pose)
        # 변위를 회전된 월드 축으로 표현: $$\Delta p_W=R_{WM}\Delta p_M$$
        displacement = world_shift[:3, :3] @ np.array([-0.02, 0, 0])
        result = plan_linear_motion(self.q_start, anchor, displacement, steps=4, solver=self.solver)
        self.assertTrue(result.success, result.message)
        # R팁의 시작 월드 자세: $${}^W T_R={}^W T_M{}^M T_R$$
        expected_start = world_shift @ self.start.T_world_tip_R
        assert_allclose(result.target_poses_world[0], expected_start, atol=1e-12)
        final = anchor.place(self.solver.fk.forward(result.q_path_rad[-1]))
        assert_allclose(final.T_world_tip_L, fixed_pose, atol=1e-12)
        assert_allclose(final.T_world_tip_R[:3, 3], expected_start[:3, 3] + displacement, atol=1e-6)
        assert_allclose(final.T_world_tip_R[:3, :3], expected_start[:3, :3], atol=1e-6)

    def test_endpoint_ik_success_does_not_hide_curved_interpolation(self) -> None:
        """도착 IK가 성공해도 중간의 관절 보간이 직선을 벗어나면 경로를 거부한다."""
        goal = self.start.T_world_tip_L.copy()
        # L팁의 도착 위치를 빔 전진 방향으로 지정: $$p_{d,x}=p_{0,x}+0.10$$
        goal[0, 3] += 0.10
        endpoint = self.solver.solve(self.anchor.to_relative_target(goal), self.q_start)
        self.assertTrue(endpoint.success, endpoint.message)
        result = plan_linear_motion(
            self.q_start, self.anchor, [0.10, 0, 0], steps=1,
            max_joint_step_rad=3.0, solver=self.solver,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.q_path_rad)
        self.assertIn("FK 오차 초과", result.message)
        self.assertGreater(result.max_position_error_m, self.solver.settings.position_tolerance_m)

    def test_late_failure_does_not_return_partial_path(self) -> None:
        """앞쪽 여러 구간이 풀린 뒤 오차가 커져도 부분 경로를 성공 결과로 내보내지 않는다."""
        result = plan_linear_motion(self.q_start, self.anchor, [0.15, 0, 0], steps=30, solver=self.solver)
        self.assertFalse(result.success)
        self.assertGreater(result.failed_step, 1)
        self.assertIsNone(result.q_path_rad)

    def test_ik_failure_and_joint_jump_are_reported(self) -> None:
        """도달할 수 없는 목표와 설정한 관절각 변화 한계 초과를 구분해 알린다."""
        unreachable = plan_linear_motion(self.q_start, self.anchor, [100, 0, 0], steps=1, solver=self.solver)
        self.assertFalse(unreachable.success)
        self.assertIsNone(unreachable.q_path_rad)
        self.assertIn("IK 실패", unreachable.message)
        jump = plan_linear_motion(
            self.q_start, self.anchor, [0.10, 0, 0],
            max_joint_step_rad=0.001, solver=self.solver,
        )
        self.assertFalse(jump.success)
        self.assertIsNone(jump.q_path_rad)
        self.assertIn("관절각 급변", jump.message)

    def test_zero_motion_and_invalid_inputs(self) -> None:
        """영 변위는 시작 자세를 유지하고 범위 밖 시작각과 잘못된 경로 입력은 거부한다."""
        result = plan_linear_motion(self.q_start, self.anchor, [0, 0, 0], steps=2, solver=self.solver)
        self.assertTrue(result.success, result.message)
        assert_allclose(result.q_path_rad, np.repeat(self.q_start[None, :], 3, axis=0))
        for displacement, steps in (([0, 0], 2), ([np.nan, 0, 0], 2), ([0, 0, 0], 0)):
            with self.subTest(displacement=displacement, steps=steps), self.assertRaises(ValueError):
                plan_linear_motion(self.q_start, self.anchor, displacement, steps=steps, solver=self.solver)
        with self.assertRaises(ValueError):
            plan_linear_motion([4.0] * 7, self.anchor, [0, 0, 0], solver=self.solver)


if __name__ == "__main__":
    unittest.main()
