"""고정턱 기준 간격과 횡이동의 방향·높이·자세 유지 조건을 검증한다."""

from dataclasses import replace
import unittest

import numpy as np

from perception.beam_edge_observation import EdgeObservation
from planning.beam_exit import BeamExitPlanner, ExitSettings


def reference_observation():
    r"""저장된 3단계와 비슷한 빔 모서리의 기하학적 시험 입력을 만든다.

    $$p_e=bu+[0,0,d]^T$$
    """
    normal = np.array([0., 0., -1.])
    outward = np.array([.9265, .3763, 0.])
    outward /= np.linalg.norm(outward)
    # 법선과 폭 축에 직교하는 길이 방향: $$a=u\times n$$
    axis = np.cross(outward, normal)
    # 모서리를 지정한 횡좌표와 광학 깊이에 배치: $$p_e=0.06735u+[0,0,0.21881]^T$$
    point = .06735 * outward + np.array([0., 0., .21881])
    return EdgeObservation(normal, .21881, .07, .0003, .01, 0., 0., 3, point, axis, outward, .0001, False)


REFERENCE_Q = np.deg2rad([-11.42578125, -50.888671875, 12.744140625, 69.521484375,
                           4.658203125, 67.763671875, 22.236328125])
GRIPPERS = {"G_L": 4.21875, "G_R": -119.8828125}


class BeamExitTests(unittest.TestCase):
    """열린 턱의 전체 폭 대신 지정한 고정턱 면을 기준으로 계획하는지 확인한다."""

    def setUp(self):
        """실물과 분리된 계획기와 참고 자세를 준비한다."""
        self.planner = BeamExitPlanner()
        self.observation = reference_observation()

    def test_fixed_lip_clearance_is_independent_of_open_jaw(self):
        """열리는 턱의 각도가 바뀌어도 고정턱 옆 간격은 바뀌지 않는다."""
        first = self.planner.measure(REFERENCE_Q, GRIPPERS, self.observation)
        self.assertAlmostEqual(first.clearance_m, -.0113, delta=.0004)
        other = self.planner.measure(REFERENCE_Q, {**GRIPPERS, "G_R": -30.}, self.observation)
        self.assertAlmostEqual(first.clearance_m, other.clearance_m, places=10)
        shifted = replace(self.observation, edge_point_m=self.observation.edge_point_m + .02 * self.observation.outward)
        self.assertAlmostEqual(self.planner.measure(REFERENCE_Q, GRIPPERS, shifted).clearance_m,
                               first.clearance_m - .02, places=10)

    def test_closed_loop_geometry_reaches_clearance_without_rising(self):
        r"""고정 공간 모서리로 다시 계산하면서 횡이동하고 팁 방향·높이를 유지한다.

        $$\Delta p_C=R_{WC}^T(p_{end}-p_{start})$$
        """
        planner, q, observation = self.planner, REFERENCE_Q.copy(), self.observation
        camera = planner.solver.fk.depth_camera_pose(q)
        reference = observation.reference().transformed(camera[:3, :3], camera[:3, 3])
        start = planner.solver.fk.forward(q).T_world_tip_R.copy()
        initial_gap = planner.measure(q, GRIPPERS, observation).vertical_gap_m
        for _ in range(12):
            current_camera = planner.solver.fk.depth_camera_pose(q)
            # 기준 모서리를 현재 카메라에서 표현할 이동량: $$t=-R^Tp$$
            translation = -current_camera[:3, :3].T @ current_camera[:3, 3]
            current = reference.transformed(current_camera[:3, :3].T, translation)
            observation = replace(observation, normal=current.normal, plane_offset_m=current.plane_offset_m,
                                  edge_point_m=current.point_m, axis=current.axis, outward=current.outward)
            measured = planner.measure(q, GRIPPERS, observation)
            self.assertAlmostEqual(measured.vertical_gap_m, initial_gap, delta=.0001)
            if measured.clearance_m >= .009:
                break
            step = planner.plan(q, GRIPPERS, observation)
            self.assertLessEqual(np.max(np.abs(np.rad2deg(step.q_rad - q))), 3.)
            q = step.q_rad
        else:
            self.fail("제한된 횡이동으로 목표 옆 간격에 도달하지 못했습니다.")
        self.assertAlmostEqual(measured.clearance_m, .010, delta=.001)
        end = planner.solver.fk.forward(q).T_world_tip_R
        np.testing.assert_allclose(end[:3, :3], start[:3, :3], atol=1e-5)
        # 초기 카메라에서 본 전체 이동량: $$\Delta p_C=R_{WC}^T(p_{end}-p_{start})$$
        displacement = camera[:3, :3].T @ (end[:3, 3] - start[:3, 3])
        self.assertGreater(displacement @ self.observation.outward, .015)
        self.assertAlmostEqual(displacement @ self.observation.normal, 0., delta=.0001)

    def test_wrong_side_and_invalid_settings_are_rejected(self):
        """방향이 뒤집힌 관측과 유효하지 않은 설정을 거부한다."""
        with self.assertRaises(ValueError):
            self.planner.measure(REFERENCE_Q, GRIPPERS, replace(self.observation, outward=-self.observation.outward))
        with self.assertRaises(ValueError):
            ExitSettings(clearance_m=.001)


if __name__ == "__main__":
    unittest.main()
