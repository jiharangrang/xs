"""초기 배치, 관절각 전달, 좌표변환과 반복 호출의 FK 계약을 검증한다."""

import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kinematics.fk import ARM_JOINT_NAMES, DEFAULT_MODEL_PATH, ForwardKinematics


class ForwardKinematicsTests(unittest.TestCase):
    """기하학적으로 알려진 자세와 별도 MuJoCo 상태를 비교한다."""

    def setUp(self) -> None:
        """각 검증에서 독립적으로 사용할 FK 계산기를 준비한다."""
        self.fk = ForwardKinematics()

    def test_initial_pose(self) -> None:
        """초기 팁 위치와 상대 축 방향이 설정한 배치와 일치하는지 확인한다."""
        result = self.fk.forward(np.zeros(7))
        assert_allclose(
            result.T_world_tip_L[:3, 3], [0, -0.03636, 0.241530662041], atol=1e-12
        )
        assert_allclose(
            result.T_world_tip_R[:3, 3], [0, 0.03636, -0.241530662041], atol=1e-12
        )
        assert_allclose(
            result.T_tip_L_tip_R[:3, 3], [0, -0.07272, -0.483061324082], atol=1e-12
        )
        assert_allclose(result.T_tip_L_tip_R[:3, :3], np.diag([1, -1, -1]), atol=1e-12)

    def test_positive_J2_rotation(self) -> None:
        r"""J2만 움직였을 때 팁이 월드 Y축의 양의 방향으로 회전하는지 확인한다.

        $$
        {}^W p_R(\theta) = a + R_y(\theta)\bigl({}^W p_R(0)-a\bigr)
        $$

        a는 초기 J2 회전축 위의 점이며, theta는 J2에 적용한 라디안 각도이다.
        """
        initial = self.fk.forward(np.zeros(7))
        theta = 0.3
        q_rad = np.array([0, theta, 0, 0, 0, 0, 0])
        # Y축 양의 회전행렬: $$R_y(\theta)=\begin{bmatrix}\cos\theta&0&\sin\theta\\0&1&0\\-\sin\theta&0&\cos\theta\end{bmatrix}$$
        rotation_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        anchor = np.array([0, 0, 0.1258])
        # 회전축에서 초기 팁까지의 변위: $$r = {}^W p_R(0) - a$$
        offset = initial.T_world_tip_R[:3, 3] - anchor
        # 회전 후 변위: $$r' = R_y(\theta)r$$
        rotated_offset = rotation_y @ offset
        # 회전 후 월드 위치: $${}^W p_R(\theta) = a + r'$$
        expected_position = anchor + rotated_offset
        # 회전 후 팁 방향: $${}^W R_R(\theta) = R_y(\theta){}^W R_R(0)$$
        expected_rotation = rotation_y @ initial.T_world_tip_R[:3, :3]

        result = self.fk.forward(q_rad)
        assert_allclose(result.T_world_tip_R[:3, 3], expected_position, atol=1e-12)
        assert_allclose(result.T_world_tip_R[:3, :3], expected_rotation, atol=1e-12)
        assert_allclose(result.T_world_tip_L, initial.T_world_tip_L, atol=1e-12)

    def test_matches_mujoco_with_finger_angles(self) -> None:
        """혼합 관절각과 손가락 개폐에서도 팁 결과가 MuJoCo와 같은지 확인한다."""
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
        data = mujoco.MjData(model)
        random = np.random.default_rng(42)
        for q_rad in random.uniform(-1.0, 1.0, size=(8, 7)):
            with self.subTest(q_rad=q_rad):
                mujoco.mj_resetData(model, data)
                for name, angle in zip(ARM_JOINT_NAMES, q_rad, strict=True):
                    data.joint(name).qpos[0] = angle
                data.joint("G_L").qpos[0] = 0.4
                data.joint("G_R").qpos[0] = -0.3
                mujoco.mj_forward(model, data)
                result = self.fk.forward(q_rad)
                for name, transform in (
                    ("tip_L", result.T_world_tip_L),
                    ("tip_R", result.T_world_tip_R),
                ):
                    assert_allclose(transform[:3, 3], data.site(name).xpos, atol=1e-12)
                    assert_allclose(
                        transform[:3, :3], data.site(name).xmat.reshape(3, 3), atol=1e-12
                    )
                    assert_array_equal(transform[3], [0, 0, 0, 1])

    def test_relative_transform_maps_points_to_world(self) -> None:
        r"""상대변환으로 옮긴 점이 월드변환으로 직접 옮긴 점과 같은지 확인한다.

        $$
        {}^W T_L\bigl({}^L T_R\,{}^R p\bigr) = {}^W T_R\,{}^R p
        $$

        p는 오른쪽 팁 좌표계의 임의 점을 동차좌표로 표현한 것이다.
        """
        result = self.fk.forward([0.2, -0.4, 0.3, 0.6, -0.5, 0.1, 0.25])
        point_tip_R = np.array([0.01, -0.02, 0.03, 1])
        # 오른쪽 팁 좌표의 점을 왼쪽 팁 좌표로 변환: $${}^L p = {}^L T_R\,{}^R p$$
        point_tip_L = result.T_tip_L_tip_R @ point_tip_R
        # 왼쪽 팁을 거쳐 월드로 변환: $${}^W p = {}^W T_L\,{}^L p$$
        world_via_tip_L = result.T_world_tip_L @ point_tip_L
        # 오른쪽 팁에서 월드로 직접 변환: $${}^W p = {}^W T_R\,{}^R p$$
        world_direct = result.T_world_tip_R @ point_tip_R
        assert_allclose(world_via_tip_L, world_direct, atol=1e-12)
        assert_array_equal(result.T_tip_L_tip_R[3], [0, 0, 0, 1])

    def test_results_do_not_share_calculation_state(self) -> None:
        """이전 결과와 입력 배열이 다음 호출 때문에 바뀌지 않는지 확인한다."""
        q_rad = np.zeros(7)
        initial = self.fk.forward(q_rad)
        saved_position = initial.T_world_tip_R.copy()
        saved_relative = initial.T_tip_L_tip_R.copy()
        self.fk.forward([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert_array_equal(initial.T_world_tip_R, saved_position)
        assert_array_equal(initial.T_tip_L_tip_R, saved_relative)
        assert_array_equal(q_rad, np.zeros(7))
        initial.T_world_tip_R[:] = 123
        repeated = self.fk.forward(q_rad)
        assert_allclose(repeated.T_world_tip_R, saved_position, atol=1e-12)

    def test_rejects_invalid_input(self) -> None:
        """잘못된 관절 수, 배열 형상, 비유한 값과 복소수를 거부하는지 확인한다."""
        invalid_inputs = (
            [], [0] * 6, [0] * 9, np.zeros((7, 1)),
            [np.nan] * 7, [np.inf] * 7, [1j] * 7,
        )
        for q_rad in invalid_inputs:
            with self.subTest(q_rad=q_rad), self.assertRaises(ValueError):
                self.fk.forward(q_rad)


if __name__ == "__main__":
    unittest.main()
