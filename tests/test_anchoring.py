"""고정 팁 전환, 월드 배치와 IK 목표 변환을 별도 MuJoCo 상태와 비교해 검증한다."""

import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from kinematics.anchoring import TipAnchor
from kinematics.fk import DEFAULT_MODEL_PATH, ForwardKinematics
from kinematics.ik import IKSettings, InverseKinematics
from kinematics.joints import ARM_JOINT_NAMES
from kinematics.poses import invert_pose


def arbitrary_world_pose() -> np.ndarray:
    """부호나 곱셈 순서 오류를 드러내도록 여러 축 회전과 평행이동이 있는 자세를 만든다."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.7]).as_matrix()
    pose[:3, 3] = [0.4, -0.2, 0.8]
    return pose


class TipAnchorTests(unittest.TestCase):
    """관절 정의를 유지한 채 어느 팁이든 지정된 월드 자세에 고정되는지 확인한다."""

    def setUp(self) -> None:
        """특이 자세를 피한 관절각과 독립적인 FK 계산기를 준비한다."""
        self.fk = ForwardKinematics()
        self.q_start = np.array([0.2, -0.4, 0.3, 0.6, -0.5, 0.1, 0.25])

    def test_inverse_matches_general_matrix_inverse(self) -> None:
        """위치 부호뿐 아니라 회전까지 뒤집는 강체 역변환을 일반 역행렬과 비교한다."""
        pose = arbitrary_world_pose()
        inverse = invert_pose(pose)
        assert_allclose(inverse, np.linalg.inv(pose), atol=1e-12)
        assert_allclose(invert_pose(inverse), pose, atol=1e-12)
        self.assertFalse(np.shares_memory(inverse, pose))

    def test_original_left_anchor_preserves_existing_fk(self) -> None:
        """XML과 같은 왼쪽 고정점을 사용하면 기존 FK의 모든 출력이 유지된다."""
        start = self.fk.forward(self.q_start)
        anchor = TipAnchor("tip_L", start.T_world_tip_L)
        for angles in (self.q_start, np.zeros(7)):
            model_fk = self.fk.forward(angles)
            placed = anchor.place(model_fk)
            assert_allclose(anchor.world_from_model(model_fk), np.eye(4), atol=1e-12)
            assert_allclose(placed.T_world_tip_L, model_fk.T_world_tip_L, atol=1e-12)
            assert_allclose(placed.T_world_tip_R, model_fk.T_world_tip_R, atol=1e-12)
            assert_array_equal(placed.T_tip_L_tip_R, model_fk.T_tip_L_tip_R)

    def test_fixed_tip_and_relocated_robot_match_mujoco(self) -> None:
        r"""로봇 뿌리의 배치를 바꾼 MuJoCo와 두 팁을 비교하고 빔과 관절각 유지를 확인한다.

        $$
        {}^W T_B={}^W T_M\,{}^M T_B
        $$

        B는 XML의 gripper_L 몸체, M은 XML 배치, W는 실제 월드이다.
        """
        fixed_pose = arbitrary_world_pose()
        for fixed_tip in ("tip_L", "tip_R"):
            anchor = TipAnchor(fixed_tip, fixed_pose)
            moving_positions = []
            for q_rad in (self.q_start, np.zeros(7)):
                with self.subTest(fixed_tip=fixed_tip, q_rad=q_rad):
                    model_fk = self.fk.forward(q_rad)
                    placed = anchor.place(model_fk)
                    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    beam_position = data.body("ibeam").xpos.copy()
                    beam_rotation = data.body("ibeam").xmat.copy()

                    root = model.body("gripper_L")
                    T_model_root = np.eye(4)
                    T_model_root[:3, 3] = root.pos
                    root_rotation = np.empty(9)
                    mujoco.mju_quat2Mat(root_rotation, root.quat)
                    T_model_root[:3, :3] = root_rotation.reshape(3, 3)
                    # 로봇 뿌리의 월드 배치: $${}^W T_B={}^W T_M\,{}^M T_B$$
                    T_world_root = anchor.world_from_model(model_fk) @ T_model_root
                    root.pos[:] = T_world_root[:3, 3]
                    mujoco.mju_mat2Quat(root.quat, T_world_root[:3, :3].reshape(-1))
                    for name, angle in zip(ARM_JOINT_NAMES, q_rad, strict=True):
                        data.joint(name).qpos[0] = angle
                    mujoco.mj_forward(model, data)

                    for name, expected in (
                        ("tip_L", placed.T_world_tip_L), ("tip_R", placed.T_world_tip_R),
                    ):
                        assert_allclose(data.site(name).xpos, expected[:3, 3], atol=1e-12)
                        assert_allclose(data.site(name).xmat.reshape(3, 3), expected[:3, :3], atol=1e-12)
                    assert_allclose(data.site(fixed_tip).xpos, fixed_pose[:3, 3], atol=1e-12)
                    assert_allclose(data.site(fixed_tip).xmat.reshape(3, 3), fixed_pose[:3, :3], atol=1e-12)
                    assert_array_equal(data.body("ibeam").xpos, beam_position)
                    assert_array_equal(data.body("ibeam").xmat, beam_rotation)
                    assert_array_equal([data.joint(name).qpos[0] for name in ARM_JOINT_NAMES], q_rad)
                    assert_array_equal(placed.T_tip_L_tip_R, model_fk.T_tip_L_tip_R)
                    moving_tip = "tip_R" if fixed_tip == "tip_L" else "tip_L"
                    moving_positions.append(data.site(moving_tip).xpos.copy())
            self.assertFalse(np.allclose(moving_positions[0], moving_positions[1]))

    def test_switching_support_preserves_pose_at_switch(self) -> None:
        """현재 관절각을 유지한 채 L 고정에서 R 고정으로 바꿔도 로봇 위치가 뛰지 않는다."""
        model_fk = self.fk.forward(self.q_start)
        left_anchor = TipAnchor("tip_L", arbitrary_world_pose())
        before = left_anchor.place(model_fk)
        right_anchor = TipAnchor("tip_R", before.T_world_tip_R)
        after = right_anchor.place(model_fk)
        assert_allclose(before.T_world_tip_L, after.T_world_tip_L, atol=1e-12)
        assert_allclose(before.T_world_tip_R, after.T_world_tip_R, atol=1e-12)
        assert_allclose(left_anchor.world_from_model(model_fk), right_anchor.world_from_model(model_fk), atol=1e-12)
        moved = right_anchor.place(self.fk.forward(np.zeros(7)))
        assert_allclose(moved.T_world_tip_R, before.T_world_tip_R, atol=1e-12)
        self.assertFalse(np.allclose(moved.T_world_tip_L, before.T_world_tip_L))

    def test_world_goal_reuses_existing_ik_for_either_fixed_tip(self) -> None:
        """어느 팁을 고정하든 월드 목표를 기존 IK에 전달해 움직이는 팁이 도착하는지 확인한다."""
        q_goal = np.array([0.22, -0.42, 0.31, 0.63, -0.52, 0.12, 0.23])
        model_goal = self.fk.forward(q_goal)
        for fixed_tip in ("tip_L", "tip_R"):
            with self.subTest(fixed_tip=fixed_tip):
                anchor = TipAnchor(fixed_tip, arbitrary_world_pose())
                placed_goal = anchor.place(model_goal)
                goal = placed_goal.T_world_tip_R if fixed_tip == "tip_L" else placed_goal.T_world_tip_L
                target = anchor.to_relative_target(goal)
                assert_allclose(target, model_goal.T_tip_L_tip_R, atol=1e-12)
                solver = InverseKinematics(settings=IKSettings(starts=1))
                result = solver.solve(target, self.q_start)
                self.assertTrue(result.success, result.message)
                actual = anchor.place(self.fk.forward(result.q_rad))
                assert_allclose(actual.T_world_tip_L, placed_goal.T_world_tip_L, atol=1e-6)
                assert_allclose(actual.T_world_tip_R, placed_goal.T_world_tip_R, atol=1e-6)

    def test_anchor_and_results_do_not_share_writable_arrays(self) -> None:
        """외부 입력이나 이전 출력의 수정으로 고정 자세와 다음 계산이 바뀌지 않는다."""
        pose = arbitrary_world_pose()
        saved_pose = pose.copy()
        anchor = TipAnchor("tip_R", pose)
        pose[:] = 0
        assert_array_equal(anchor.T_world_fixed_tip, saved_pose)
        with self.assertRaises(ValueError):
            anchor.T_world_fixed_tip[0, 3] = 123
        model_fk = self.fk.forward(self.q_start)
        placed = anchor.place(model_fk)
        placed.T_world_tip_R[:] = 0
        placed.T_tip_L_tip_R[:] = 0
        repeated = anchor.place(model_fk)
        assert_allclose(repeated.T_world_tip_R, saved_pose, atol=1e-12)
        assert_allclose(repeated.T_tip_L_tip_R, self.fk.forward(self.q_start).T_tip_L_tip_R, atol=1e-12)

    def test_rejects_unknown_tip_and_invalid_transforms(self) -> None:
        """팁 이름 오타와 잘못된 고정 자세·목표 자세를 조기에 거부한다."""
        with self.assertRaises(ValueError):
            TipAnchor("grip_R", np.eye(4))
        anchor = TipAnchor("tip_L", np.eye(4))
        for invalid in (np.eye(3), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    TipAnchor("tip_R", invalid)
                with self.assertRaises(ValueError):
                    anchor.to_relative_target(invalid)
                with self.assertRaises(ValueError):
                    invert_pose(invalid)


if __name__ == "__main__":
    unittest.main()
