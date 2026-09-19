"""짧아진 몸에서 앞발을 전진시키며 높이·정면·횡위치를 보정하는 기하학을 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from perception.beam_edge_observation import EdgeObservation
from planning.front_advance import FrontAdvancePlanner
from planning.linear_motion import LinearPathResult, plan_linear_motion
from planning.rear_pull import RearPullPlanner
from simulation.model import load_model, set_path_time
from test_rear_pull import Q_DEG, GRIPPERS, AXIS


def sampled_preflight_case():
    """실물 실패 직후 저장한 관절각과 세 프레임의 빔 관측을 재현한다."""
    record = json.loads((Path(__file__).parent / "fixtures/stage8_fk_sampling.json").read_text())
    observed = record["observation"]
    for name in ("normal", "edge_point_m", "axis", "outward"):
        observed[name] = np.array(observed[name])
    return np.array(record["q_rad"]), record["grippers_deg"], EdgeObservation(**observed)


def compact_pose():
    """기존 7단계 모델 경로의 당김·횡복귀 끝 자세를 반환한다."""
    return RearPullPlanner().plan(np.deg2rad(Q_DEG), GRIPPERS, AXIS).side_return.q_rad[-1].copy()


def grasp_observation(planner, q):
    r"""기존 높이 목표에 있고 고정턱이 빔 안으로 걸쳐진 시험 관측을 만든다.

    $$p_e=(\min_i u^Tp_{lip,i}+0.015)u+[0,0,d]^T$$
    """
    planner.geometry.update(q, GRIPPERS)
    normal = np.array([0., 0., -1.])
    outward = planner.geometry.outward_camera()
    outward[2] = 0.
    # 앞 고정턱 쪽의 단위 횡방향: $$u=\tilde u/\|\tilde u\|$$
    outward /= np.linalg.norm(outward)
    _, depth = planner.height.height_error(q, normal, .2)
    # 처음 고정턱은 빔 안으로 15 mm 겹치도록 구성: $$b=\min_i u^Tp_{lip,i}+0.015$$
    edge = float(np.min(planner.geometry.points_camera(fixed_lip=True) @ outward)) + .015
    # 평면 위 모서리 위치: $$p_e=bu+[0,0,d]^T$$
    point = edge * outward + np.array([0., 0., depth])
    # 빔 평면에서 횡방향에 수직인 길이 방향: $$a=u\times n$$
    axis = np.cross(outward, normal)
    return EdgeObservation(normal, depth, .07, .0003, .01, 0., 0., 3, point, axis, outward, .0001, False)


def observation_from_world(planner, reference, q):
    r"""고정된 빔을 이동한 카메라 좌표로 다시 관측한다.

    $$T_{CW}=[R_{WC}^T,-R_{WC}^Tp_{WC}]$$
    """
    camera = planner.solver.fk.depth_camera_pose(q)
    # 현재 카메라 변환의 역이동: $$t_{CW}=-R_{WC}^Tp_{WC}$$
    translation = -camera[:3, :3].T @ camera[:3, 3]
    current = reference.transformed(camera[:3, :3].T, translation)
    return EdgeObservation(current.normal, current.plane_offset_m, current.width_m, .0003, .01,
                           0., 0., 3, current.point_m, current.axis, current.outward, .0001, True)


class FrontAdvanceTests(unittest.TestCase):
    """개방·직진 중 앞발의 목표와 뒷발 고정을 별도로 확인한다."""

    def setUp(self):
        """7단계 후 압축된 자세와 고정된 시험 빔을 준비한다."""
        self.q = compact_pose()
        self.planner = FrontAdvancePlanner()
        self.observed = grasp_observation(self.planner, self.q)
        camera = self.planner.solver.fk.depth_camera_pose(self.q)
        self.beam = self.observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        self.planner.configure(self.q, GRIPPERS, self.observed)

    def test_full_preview_keeps_rear_fixed_and_front_on_straight_line(self):
        """MuJoCo에서 뒷발을 유지하고 앞발을 옆으로 빼지 않은 채 100 mm 직진한다."""
        path = self.planner.preview(self.q, GRIPPERS, self.observed)
        self.assertEqual([s.name for s in path.segments],
                         ["front_open", "front_advance", "front_close"])
        model, data = load_model(self.q)
        set_path_time(model, data, path, 0.)
        rear = data.site("tip_L").xpos.copy()
        start = data.site("tip_R").xpos.copy()
        for elapsed in np.linspace(0., path.duration_s, 40):
            set_path_time(model, data, path, elapsed)
            np.testing.assert_allclose(data.site("tip_L").xpos, rear, atol=1e-12)
            self.assertAlmostEqual(float((data.site("tip_R").xpos - start) @ self.planner.reference.outward_world),
                                   0., delta=.0001)
        delta = data.site("tip_R").xpos - start
        self.assertAlmostEqual(float(delta @ self.planner.reference.axis_world), .1, delta=.0001)
        self.assertAlmostEqual(float(delta @ self.planner.reference.outward_world), 0., delta=.0001)
        for segment in path.segments:
            np.testing.assert_allclose(np.rad2deg(segment.gripper_q_rad[:, 0]), GRIPPERS["G_L"])
        self.assertAlmostEqual(np.rad2deg(path.segments[-1].gripper_q_rad[-1, 1]), 4.6)
        path.validate_limits()

    def test_observed_loop_keeps_starting_line_without_seeking_side_clearance(self):
        """고정턱이 빔 안에 겹쳐 있어도 옆으로 빼지 않고 높이를 유지하며 직진한다."""
        q = self.q.copy()
        for _ in range(120):
            observed = observation_from_world(self.planner, self.beam, q)
            measured = self.planner.measure(q, GRIPPERS, observed)
            self.assertLess(abs(measured.lateral_error_m), .001)
            self.assertLess(measured.side_clearance_m, 0.)
            if abs(measured.forward_error_m) <= .001 and abs(measured.height_error_m) <= .0005:
                break
            q = self.planner.plan(q, GRIPPERS, observed).q_rad
        else:
            self.fail("앞발 직진이 수렴하지 않았습니다.")
        self.assertAlmostEqual(measured.progress_m, .1, delta=.001)
        self.assertLess(abs(measured.height_error_m), .0005)

    def test_forward_height_and_tilt_are_corrected_in_one_goal(self):
        """전진 중 높이와 기울기 오차도 같은 관절 목표에서 함께 줄인다."""
        normal = np.array([.02, -.015, -1.])
        normal /= np.linalg.norm(normal)
        observed = replace(self.observed, normal=normal, plane_offset_m=self.observed.plane_offset_m + .004)
        camera = self.planner.solver.fk.depth_camera_pose(self.q)
        beam = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        before = self.planner.measure(self.q, GRIPPERS, observed)
        step = self.planner.plan(self.q, GRIPPERS, observed)
        after = self.planner.measure(step.q_rad, GRIPPERS, observation_from_world(self.planner, beam, step.q_rad))
        self.assertGreater(after.progress_m, before.progress_m)
        self.assertLess(abs(after.height_error_m), abs(before.height_error_m))
        self.assertLess(after.tilt_deg, before.tilt_deg)

    def test_observed_edge_shift_does_not_command_side_exit(self):
        """검출 모서리가 흔들려 옆 여유가 달라져도 불필요한 횡이동을 만들지 않는다."""
        expected = self.planner.plan(self.q, GRIPPERS, self.observed)
        for offset in (-.03, .03):
            observed = replace(self.observed, edge_point_m=self.observed.edge_point_m + offset * self.observed.outward)
            actual = self.planner.plan(self.q, GRIPPERS, observed)
            self.assertEqual(actual.lateral_m, 0.)
            np.testing.assert_allclose(actual.q_rad, expected.q_rad, atol=1e-12)

    def test_reversed_beam_axis_keeps_forward_direction(self):
        """검출 선의 양끝이 뒤집혀도 뒷발에서 앞발 쪽으로 전진한다."""
        direction = self.planner.reference.axis_world.copy()
        self.planner.configure(self.q, GRIPPERS, replace(self.observed, axis=-self.observed.axis))
        np.testing.assert_allclose(self.planner.reference.axis_world, direction)

    def test_invalid_distance_and_axis_are_rejected(self):
        """잘못된 거리나 빔 방향으로 개방 전 경로를 생성하지 않는다."""
        for distance in (0., -1., 101., np.nan, np.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.planner.configure(self.q, GRIPPERS, self.observed, distance)
        with self.assertRaises(ValueError):
            self.planner.configure(self.q, GRIPPERS, replace(self.observed, axis=[0., 0., 1.]))

    def test_real_pose_straight_preview_preserves_full_distance_and_limits(self):
        """기존 실물 실패 자세에서도 허용오차를 지키며 전체 직진 경로를 검사한다."""
        q, grips, observed = sampled_preflight_case()
        self.planner.configure(q, grips, observed)
        calls = []

        def record(*args, **kwargs):
            """재시도 간격과 실제 IK·FK 검사 결과를 기록한다."""
            result = plan_linear_motion(*args, **kwargs)
            calls.append((kwargs["steps"], result))
            return result

        with patch("planning.front_advance.plan_linear_motion", side_effect=record):
            path = self.planner.preview(q, grips, observed)
        self.assertEqual([s.name for s in path.segments], ["front_open", "front_advance", "front_close"])
        settings = self.planner.solver.settings
        self.assertEqual(settings.position_tolerance_m, .0001)
        self.assertEqual(settings.rotation_tolerance_rad, .001)
        for _, result in calls:
            if result.success:
                self.assertLessEqual(result.max_position_error_m, settings.position_tolerance_m)
                self.assertLessEqual(result.max_rotation_error_rad, settings.rotation_tolerance_rad)
        end = self.planner.solver.fk.forward(path.segments[-1].q_rad[-1]).T_world_tip_R[:3, 3]
        delta = end - self.planner.reference.point_world_m
        self.assertAlmostEqual(float(delta @ self.planner.reference.axis_world), .1, delta=.0001)
        self.assertAlmostEqual(float(delta @ self.planner.reference.outward_world), 0., delta=.0001)
        path.validate_limits()

    def test_ik_failure_is_not_retried_as_sampling_error(self):
        """IK 해를 찾지 못한 경우에는 보간 간격 문제로 간주하지 않는다."""
        result = LinearPathResult(False, None, np.zeros((2, 4, 4)), failed_step=1, message="시험 IK 실패")
        with patch("planning.front_advance.plan_linear_motion", return_value=result) as solve:
            with self.assertRaisesRegex(ValueError, "IK 실패"):
                self.planner.preview(self.q, GRIPPERS, self.observed)
        self.assertEqual(solve.call_count, 1)

    def test_sampling_refinement_stops_after_four_attempts(self):
        """계속 실패하는 보간 검사 때문에 무제한으로 계획을 반복하지 않는다."""
        result = LinearPathResult(False, None, np.zeros((2, 4, 4)), max_position_error_m=.001,
                                  failed_step=1, message="시험 FK 오차")
        with patch("planning.front_advance.plan_linear_motion", return_value=result) as solve:
            with self.assertRaisesRegex(ValueError, "FK 오차"):
                self.planner.preview(self.q, GRIPPERS, self.observed)
        self.assertEqual(solve.call_count, 4)
        steps = [call.kwargs["steps"] for call in solve.call_args_list]
        self.assertEqual(steps, [steps[0], 2 * steps[0], 4 * steps[0], 8 * steps[0]])
