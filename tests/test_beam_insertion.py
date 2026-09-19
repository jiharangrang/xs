"""4단계 복귀 기준과 실물 관측으로 횡방향·높이·정면 보정을 검증한다."""

from dataclasses import replace
import unittest

import numpy as np

from planning.beam_alignment import tilt_degrees
from planning.beam_insertion import BeamInsertionPlanner, InsertionSettings
from planning.insertion_reference import LateralReference
from test_insertion_height import (CONFIRMED_Q, TILTED_Q, GRIPPERS, confirmed_height_observation,
                                   tilted_height_observation, observation_at)


def return_reference(planner, q, observed, distance=.024):
    r"""알려진 횡복귀 거리와 다른 높이를 가진 출발 위치를 시험 기준으로 만든다.

    $$p_*=p_{tip}-su_W+0.045n_W$$
    """
    camera = planner.solver.fk.depth_camera_pose(q)
    tip = planner.solver.fk.forward(q).T_world_tip_R[:3, 3]
    # 관측 횡방향의 월드 회전: $$u_W=R_{WC}u_C$$
    outward = camera[:3, :3] @ observed.outward
    # 관측 법선의 월드 회전: $$n_W=R_{WC}n_C$$
    normal = camera[:3, :3] @ observed.normal
    # 삽입 완료 위치보다 아래에 있는 출발 위치: $$p_*=p_{tip}-su_W+0.045n_W$$
    point = tip - distance * outward + .045 * normal
    return LateralReference(point, outward, "stage4-test-run")


class BeamInsertionTests(unittest.TestCase):
    """현재 관절 위치에서 4단계 출발 횡위치까지의 오차로 삽입하는지 확인한다."""

    def setUp(self):
        """실제 장치 없이 저장된 실물 도착 자세와 공통 기구학을 준비한다."""
        self.planner = BeamInsertionPlanner()
        self.q = CONFIRMED_Q.copy()
        self.observation = confirmed_height_observation()
        self.planner.reference = return_reference(self.planner, self.q, self.observation)

    def test_saved_pose_returns_full_lateral_distance_instead_of_rgb_center(self):
        """RGB 중앙까지의 짧은 거리 대신 저장한 출발 횡위치까지 복귀한다."""
        measured = self.planner.measure(self.q, GRIPPERS, self.observation)
        self.assertAlmostEqual(measured.remaining_m, .024, places=8)
        self.assertAlmostEqual(measured.height_error_m, 0.)
        self.assertAlmostEqual(measured.overlap_m, -.01138090229, places=8)
        plane_error = self.observation.normal @ measured.rgb_center_camera_m + self.observation.plane_offset_m
        self.assertAlmostEqual(plane_error, 0., places=10)
        depth_center_error = -self.observation.outward @ self.observation.edge_point_m + self.observation.width_m / 2
        self.assertLess(depth_center_error, 0.)
        self.assertGreater(measured.remaining_m, 0.)
        changed = replace(self.observation, width_m=.05,
                          edge_point_m=self.observation.edge_point_m + .01 * self.observation.outward)
        self.assertAlmostEqual(self.planner.measure(self.q, GRIPPERS, changed).remaining_m, .024)

    def test_fresh_edge_loop_enters_beam_preserving_height_and_longitudinal_position(self):
        r"""모서리를 재관측하며 출발 횡위치로 복귀하고 길이 방향과 높이를 유지한다.

        $$\Delta p_C=R_{WC}^T(p_{end}-p_{start})$$
        """
        q, observed, planner = self.q.copy(), self.observation, self.planner
        camera = planner.solver.fk.depth_camera_pose(q)
        start = planner.solver.fk.forward(q).T_world_tip_R.copy()
        for _ in range(20):
            current = observation_at(planner, observed, self.q, q)
            measured = planner.measure(q, GRIPPERS, current)
            self.assertAlmostEqual(measured.height_error_m, 0., delta=.0001)
            if abs(measured.remaining_m) <= planner.settings.tolerance_m:
                break
            step = planner.plan(q, GRIPPERS, current)
            self.assertGreater(step.lateral_distance_m, 0.)
            self.assertLessEqual(step.lateral_distance_m, .003)
            q = step.q_rad
        else:
            self.fail("4단계 출발 횡위치로 수렴하지 못했습니다.")
        self.assertGreater(measured.overlap_m, .010)
        end = planner.solver.fk.forward(q).T_world_tip_R
        # 초기 카메라에서 본 실제 기구학 변위: $$\Delta p_C=R_{WC}^T(p_{end}-p_{start})$$
        displacement = camera[:3, :3].T @ (end[:3, 3] - start[:3, 3])
        self.assertAlmostEqual(displacement @ observed.outward, -.024, delta=.001)
        self.assertAlmostEqual(displacement @ observed.axis, 0., delta=.0001)
        self.assertAlmostEqual(displacement @ observed.normal, 0., delta=.0001)

    def test_single_edge_and_reversed_line_direction_keep_same_inward_goal(self):
        """반대 모서리가 가려져도 확인된 폭과 바깥쪽 모서리로 같은 목표를 만든다."""
        original = self.planner.plan(self.q, GRIPPERS, self.observation)
        one_edge = replace(self.observation, single_edge=True, axis=-self.observation.axis)
        candidate = self.planner.plan(self.q, GRIPPERS, one_edge)
        np.testing.assert_allclose(original.q_rad, candidate.q_rad, atol=1e-9)
        self.assertAlmostEqual(original.lateral_distance_m, candidate.lateral_distance_m)

    def test_lateral_height_and_frontal_error_share_one_target(self):
        """삽입 도중 높이와 정면이 틀어지면 같은 목표에서 세 성분을 함께 보정한다."""
        observed = tilted_height_observation()
        before = self.planner.measure(TILTED_Q, GRIPPERS, observed)
        step = self.planner.plan(TILTED_Q, GRIPPERS, observed)
        shifted = observation_at(self.planner, observed, TILTED_Q, step.q_rad)
        after = self.planner.measure(step.q_rad, GRIPPERS, shifted)
        self.assertGreater(step.lateral_distance_m, 0.)
        self.assertGreater(step.distance_m, 0.)
        self.assertLess(abs(after.remaining_m), abs(before.remaining_m))
        self.assertLess(abs(after.height_error_m), abs(before.height_error_m))
        self.assertLess(tilt_degrees(shifted.normal), tilt_degrees(observed.normal))

    def test_initial_height_alignment_can_wait_to_insert_without_failing(self):
        r"""삽입 준비 전에는 횡이동을 보류하고 높이·정면만 계속 맞춘다.

        $$\Delta p_C=R_{WC}^T(p_{next}-p_{current})$$
        """
        observed = tilted_height_observation()
        camera = self.planner.solver.fk.depth_camera_pose(TILTED_Q)
        before = self.planner.solver.fk.forward(TILTED_Q).T_world_tip_R[:3, 3].copy()
        step = self.planner.plan(TILTED_Q, GRIPPERS, observed, insert_enabled=False)
        after = self.planner.solver.fk.forward(step.q_rad).T_world_tip_R[:3, 3]
        # 현재 카메라에서 본 보정 변위: $$\Delta p_C=R_{WC}^T(p_{next}-p_{current})$$
        displacement = camera[:3, :3].T @ (after - before)
        self.assertEqual(step.lateral_distance_m, 0.)
        self.assertGreater(step.distance_m, 0.)
        self.assertAlmostEqual(displacement @ observed.outward, 0., delta=.0001)

    def test_overshot_return_position_corrects_outward(self):
        """출발 횡위치를 지난 경우 되돌아갈 거리만 반대쪽으로 보정한다."""
        self.planner.reference = return_reference(self.planner, self.q, self.observation, distance=-.003)
        before = self.planner.measure(self.q, GRIPPERS, self.observation)
        step = self.planner.plan(self.q, GRIPPERS, self.observation)
        after_observation = observation_at(self.planner, self.observation, self.q, step.q_rad)
        after = self.planner.measure(step.q_rad, GRIPPERS, after_observation)
        self.assertLess(step.lateral_distance_m, 0.)
        self.assertLess(abs(after.remaining_m), abs(before.remaining_m))

    def test_reference_is_required_without_rgb_center_fallback(self):
        """복귀 기준이 없으면 이전 RGB 중앙 목표로 대신 이동하지 않는다."""
        self.planner.reference = None
        with self.assertRaisesRegex(ValueError, "4단계"):
            self.planner.plan(self.q, GRIPPERS, self.observation)

    def test_invalid_geometry_and_settings_do_not_create_targets(self):
        """방향이 반대이거나 깊이·폭이 없을 때 임의의 삽입 명령을 만들지 않는다."""
        for observed in (replace(self.observation, outward=-self.observation.outward),
                         replace(self.observation, width_m=float("nan")),
                         replace(self.observation, plane_offset_m=-.1)):
            with self.assertRaises(ValueError):
                self.planner.plan(self.q, GRIPPERS, observed)
        with self.assertRaises(ValueError):
            InsertionSettings(tolerance_m=0.)


if __name__ == "__main__":
    unittest.main()
