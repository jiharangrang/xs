"""높이 기준과 높이·카메라 정면을 함께 보정하는 관절 목표를 검증한다."""

from dataclasses import replace
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from kinematics.mesh_sections import strip_vertices
from perception.beam_edge_observation import EdgeObservation
from planning.beam_alignment import tilt_degrees
from planning.insertion_height import InsertionHeightPlanner, InsertionHeightSettings, load_target_depth
from planning.height_reference import DEFAULT_DEPTH_REFERENCE_PATH, load_height_reference
from planning.vertical_motion import solve_height_alignment_target
from test_beam_exit import GRIPPERS


HEIGHT_Q = np.deg2rad([-5.185546875, -50.9765625, 11.6015625, 68.115234375,
                       4.21875, 69.521484375, 16.083984375])

NARROW_Q = np.deg2rad([-6.15234375, -52.55859375, 10.458984375, 74.267578125,
                       3.955078125, 62.9296875, 15.29296875])

TILTED_Q = np.deg2rad([-4.306640625, -52.119140625, 12.568359375, 70.927734375,
                       5.44921875, 63.544921875, 14.58984375])


CONFIRMED_Q = np.deg2rad([-7.119140625, -57.216796875, 8.173828125, 81.650390625,
                          3.076171875, 50.625, 14.765625])


def confirmed_height_observation():
    """사용자가 도착으로 확인한 자세의 원본 깊이 열다섯 장에서 얻은 관측을 재현한다."""
    return EdgeObservation(
        np.array([.004563989140494959, .014153711104104485, -.9998894151180455]),
        .1716460330287304, .06976247516656543, .0002930286713349457, .0075197038600805345,
        0., 0., 15, np.array([.04285563506772572, -.012572780594278101, .17168305232838363]),
        np.array([-.2622356474080215, .964923400681129, .012461783438225408]),
        np.array([.9649930752233962, .2621497726656254, .00811550751672908]),
        3.183296807624157e-05, False,
    )


CONTACT_Q = np.deg2rad([-6.064453125, -63.896484375, 9.931640625, 70.3125,
                        4.921875, 54.052734375, 15.1171875])


def contact_height_observation():
    """접촉했는데 높이가 부족하다고 판단한 화면의 실제 관측을 재현한다."""
    values = {'normal': [0.003783486748514731, -0.022661414405073722, -0.9997360379246041], 'plane_offset_m': 0.1736395254241326, 'width_m': 0.06973018247907532, 'plane_rms_m': 0.0003036322322778191, 'normal_spread_deg': 0.004331704966353054, 'first_frame_s': 472302.49980675, 'last_frame_s': 472302.700065625, 'frames': 3, 'edge_point_m': [0.04523090559176952, -0.013212770366308963, 0.1741553938177843], 'axis': [-0.2655107100222163, 0.9638370334960057, -0.022852477439393885], 'outward': [0.9641004865337377, 0.2655270873097218, -0.0023701835848389187], 'edge_spread_m': 1.8474299028003354e-05, 'single_edge': False}
    return EdgeObservation(**{key: np.array(value) if isinstance(value, list) else value for key, value in values.items()})


def flat_goal_depth(planner, q_rad=HEIGHT_Q):
    """정면을 바라보는 시험 자세에서 실측 높이 기준에 해당하는 카메라 거리를 반환한다."""
    return planner.height_error(q_rad, [0., 0., -1.], planner.settings.target_depth_m)[1]


def tilted_height_observation():
    """실물 상승 중 기울기가 누적되어 삽입 구간이 사라진 마지막 관측을 재현한다."""
    return EdgeObservation(
        np.array([.001151459181268447, -.0481217560207882, -.9988408135129588]),
        .21465785972504844, .06983508460043887, .0003251157893994539, .05501263406878301,
        0., 0., 3, np.array([.04574097947064248, -.009646813503785186, .2154968464652649]),
        np.array([-.27651953807025276, .9598795777253423, -.04656330455662023]),
        np.array([.9610076062710644, .2762526161028645, -.012201343593313995]),
        .0003127127490902634, False,
    )


def narrow_height_observation():
    """실물에서 여유 구간이 남아 있는데 중지됐던 관측을 재현한다."""
    return EdgeObservation(
        np.array([.0016504214094568816, .01453464808544363, -.9998930043330655]),
        .20060229926534898, .06981860549675405, .000319095941120596, .0042746320090240545,
        0., 0., 3, np.array([.04665268507460366, -.015558592542810312, .20047405434001386]),
        np.array([-.2763972912337715, .9609484650778602, .013512396646711635]),
        np.array([.961042045686003, .27634541677253893, .005603307280001986]),
        1.9915588624665555e-5, False,
    )


def height_observation(planner, q_rad=HEIGHT_Q, depth=.22):
    r"""고정턱이 빔 밖에 있는 수평 빔의 기하학적 시험 관측을 만든다.

    $$p_e=(\min_i u^Tp_{lip,i}-0.010)u+[0,0,d]^T$$
    """
    planner.geometry.update(q_rad, GRIPPERS)
    normal = np.array([0., 0., -1.])
    outward = planner.geometry.outward_camera()
    outward[2] = 0.
    # 수평 폭 방향의 단위 벡터: $$u=u'/\|u'\|$$
    outward /= np.linalg.norm(outward)
    # 고정턱과 빔 모서리 사이의 알려진 시험 간격: $$b=\min_i u^Tp_{lip,i}-0.010$$
    edge = float(np.min(planner.geometry.points_camera(fixed_lip=True) @ outward)) - .010
    # 알려진 모서리의 카메라 위치: $$p_e=bu+[0,0,d]^T$$
    point = edge * outward + np.array([0., 0., depth])
    # 빔 평면 위 길이 방향: $$a=u\times n$$
    axis = np.cross(outward, normal)
    return EdgeObservation(normal, depth, .07, .0003, .01, 0., 0., 3, point, axis, outward, .0001, False)


def observation_at(planner, observed, q_start, q_target):
    r"""공간에 고정된 빔을 다른 관절 자세의 카메라 좌표로 옮긴다.

    $$t_{CW}=-R_{WC}^Tp_{WC}$$
    """
    camera = planner.solver.fk.depth_camera_pose(q_start)
    world = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
    camera = planner.solver.fk.depth_camera_pose(q_target)
    # 이동 후 카메라 역변환의 이동량: $$t_{CW}=-R_{WC}^Tp_{WC}$$
    translation = -camera[:3, :3].T @ camera[:3, 3]
    current = world.transformed(camera[:3, :3].T, translation)
    return replace(observed, normal=current.normal, plane_offset_m=current.plane_offset_m,
                   edge_point_m=current.point_m, axis=current.axis, outward=current.outward)


class InsertionHeightTests(unittest.TestCase):
    """기존 높이 기준을 유지하면서 높이와 정면을 동시에 추종하는지 확인한다."""

    def setUp(self):
        """실물 장치와 분리된 CAD 계획기와 알려진 빔을 준비한다."""
        self.planner = InsertionHeightPlanner()
        self.observation = height_observation(self.planner)

    def test_triangle_crossing_beam_is_not_missed_by_vertex_filter(self):
        """원래 꼭짓점이 모두 폭 밖이어도 빔을 가로지르는 삼각형을 남긴다."""
        triangles = np.array([[[-2., 0., 1.], [2., 0., 1.], [2., 1., 1.]]])
        result = strip_vertices(triangles, np.array([1., 0., 0.]), -.2, .2)
        self.assertGreater(len(result), 0)
        self.assertTrue(np.all(np.abs(result[:, 0]) <= .2 + 1e-12))
        self.assertAlmostEqual(result[:, 0].min(), -.2)
        self.assertAlmostEqual(result[:, 0].max(), .2)

    def test_height_reference_uses_saved_depth_and_keeps_cad_diagnostics(self):
        """높이 목표는 저장한 실측 깊이로 계산하고 기존 CAD 여유는 표시용으로 남긴다."""
        measured = self.planner.measure(HEIGHT_Q, GRIPPERS, self.observation)
        self.assertAlmostEqual(measured.upper_clearance_m, -.04405, delta=.00002)
        self.assertAlmostEqual(measured.lower_clearance_m, .0478, delta=.00002)
        self.assertAlmostEqual(measured.remaining_m, .04734162032619954)
        self.assertAlmostEqual(measured.target_depth_m, .17265837967380046)
        self.assertEqual(measured.depth_m, .22)

    def test_coarse_and_fine_lift_preserve_aligned_camera(self):
        """정면이 맞으면 기존 방향과 옆 간격을 유지하며 높이 오차만 줄인다."""
        for depth, fine, limit in ((.22, False, .005), (self.planner.settings.target_depth_m + .0039, True, .0005)):
            with self.subTest(depth=depth):
                observed = height_observation(self.planner, depth=depth)
                before = self.planner.measure(HEIGHT_Q, GRIPPERS, observed)
                step = self.planner.plan(HEIGHT_Q, GRIPPERS, observed)
                shifted = observation_at(self.planner, observed, HEIGHT_Q, step.q_rad)
                after = self.planner.measure(step.q_rad, GRIPPERS, shifted)
                self.assertEqual(step.fine, fine)
                self.assertGreater(step.distance_m, 0.)
                self.assertLessEqual(step.distance_m, limit)
                self.assertLess(tilt_degrees(shifted.normal), .001)
                self.assertAlmostEqual(after.remaining_m, before.remaining_m - step.distance_m, delta=.00005)
                self.assertAlmostEqual(after.side_clearance_m, .010, delta=.00005)

    def test_one_target_both_lifts_and_corrects_real_tilt(self):
        r"""실물 실패 관측에서 한 목표가 팁을 올리는 동시에 카메라 기울기를 줄인다.

        $$\Delta p_R=-sR_Cn$$
        """
        observed = tilted_height_observation()
        before = self.planner.measure(TILTED_Q, GRIPPERS, observed)
        fk = self.planner.solver.fk
        camera = fk.depth_camera_pose(TILTED_Q)
        tip = fk.forward(TILTED_Q).T_world_tip_R[:3, 3].copy()
        step = self.planner.plan(TILTED_Q, GRIPPERS, observed)
        shifted = observation_at(self.planner, observed, TILTED_Q, step.q_rad)
        after = self.planner.measure(step.q_rad, GRIPPERS, shifted)
        # 같은 목표에 포함된 법선 방향 상승량: $$\Delta p_R=-sR_Cn$$
        displacement = -step.distance_m * (camera[:3, :3] @ observed.normal)
        np.testing.assert_allclose(fk.forward(step.q_rad).T_world_tip_R[:3, 3], tip + displacement, atol=1e-5)
        self.assertEqual(step.kind, "lift")
        self.assertGreater(step.distance_m, 0.)
        self.assertLess(after.remaining_m, before.remaining_m)
        self.assertLess(tilt_degrees(shifted.normal), tilt_degrees(observed.normal) / 2)

    def test_height_reached_does_not_plan_alignment_only(self):
        r"""높이가 맞으면 정면 오차가 남아도 추가 이동 계획을 만들지 않는다.

        $$d'=d-h,\quad p_e'=p_e+hn$$
        """
        observed = tilted_height_observation()
        height = self.planner.measure(TILTED_Q, GRIPPERS, observed).remaining_m
        # 목표 높이에 빔 평면 배치: $$d'=d-h$$
        offset = observed.plane_offset_m - height
        # 같은 평면의 모서리도 함께 이동: $$p_e'=p_e+hn$$
        point = observed.edge_point_m + height * observed.normal
        near = replace(observed, plane_offset_m=offset, edge_point_m=point)
        self.assertAlmostEqual(self.planner.measure(TILTED_Q, GRIPPERS, near).remaining_m, 0.)
        self.assertGreater(tilt_degrees(near.normal), 1.)
        with self.assertRaisesRegex(ValueError, "목표 높이에 도달"):
            self.planner.plan(TILTED_Q, GRIPPERS, near)

    def test_camera_rotation_at_fixed_tip_does_not_change_height_error(self):
        """카메라가 회전해 원점 거리가 달라져도 같은 그리퍼 지점 높이는 달라지지 않는다."""
        observed = tilted_height_observation()
        before = self.planner.measure(TILTED_Q, GRIPPERS, observed)
        _, rotation, _ = self.planner.correction(0., observed.normal)
        target, _ = solve_height_alignment_target(TILTED_Q, observed.normal, 0., rotation, solver=self.planner.solver)
        shifted = observation_at(self.planner, observed, TILTED_Q, target)
        after = self.planner.measure(target, GRIPPERS, shifted)
        self.assertGreater(abs(before.depth_m - after.depth_m), .0005)
        self.assertAlmostEqual(before.remaining_m, after.remaining_m, delta=.00001)

    def test_contact_screenshot_reaches_height_without_false_rise(self):
        """접촉 화면의 원점 거리 차이를 추가 상승량으로 오인하지 않는다."""
        observed = contact_height_observation()
        measured = self.planner.measure(CONTACT_Q, GRIPPERS, observed)
        self.assertAlmostEqual(observed.plane_offset_m - self.planner.settings.target_depth_m, .0019934923954)
        self.assertAlmostEqual(measured.remaining_m, -.000342318674, places=9)
        self.assertGreater(tilt_degrees(observed.normal), 1.)
        with self.assertRaisesRegex(ValueError, "목표 높이에 도달"):
            self.planner.plan(CONTACT_Q, GRIPPERS, observed)

    def test_overshot_height_is_corrected_downward(self):
        """높이를 지나쳤으면 실패시키지 않고 목표를 향해 낮춘다."""
        observed = height_observation(self.planner, depth=self.planner.settings.target_depth_m - .0014)
        before = self.planner.measure(HEIGHT_Q, GRIPPERS, observed)
        step = self.planner.plan(HEIGHT_Q, GRIPPERS, observed)
        shifted = observation_at(self.planner, observed, HEIGHT_Q, step.q_rad)
        after = self.planner.measure(step.q_rad, GRIPPERS, shifted)
        self.assertEqual(step.kind, "lower")
        self.assertLess(step.distance_m, 0.)
        self.assertLess(abs(after.remaining_m), abs(before.remaining_m))

    def test_display_clearances_do_not_gate_height_correction(self):
        """예전의 옆 여유·삽입 구간 폭 조건으로 높이 보정을 막지 않는다."""
        planner = InsertionHeightPlanner(settings=InsertionHeightSettings(flange_thickness_m=.020))
        observed = replace(self.observation, edge_point_m=self.observation.edge_point_m + .008 * self.observation.outward)
        measured = planner.measure(HEIGHT_Q, GRIPPERS, observed)
        self.assertLess(measured.side_clearance_m, .003)
        self.assertLess(measured.upper_clearance_m + measured.lower_clearance_m, 0.)
        self.assertGreater(planner.plan(HEIGHT_Q, GRIPPERS, observed).distance_m, 0.)

    def test_real_narrow_reference_preserves_height_goal(self):
        """좁은 실물 관측에서도 동일한 목표 중앙을 향한 상승을 만든다."""
        observed = narrow_height_observation()
        measured = self.planner.measure(NARROW_Q, GRIPPERS, observed)
        self.assertAlmostEqual(measured.remaining_m, observed.plane_offset_m - measured.target_depth_m)
        self.assertGreater(self.planner.plan(NARROW_Q, GRIPPERS, observed).distance_m, 0.)

    def test_invalid_measurements_settings_and_joint_limits_are_rejected(self):
        """계산 불가능한 깊이·설정·관절 범위에서는 유효한 명령을 만들지 않는다."""
        for depth in (float("nan"), float("inf"), 0.):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.planner.plan(HEIGHT_Q, GRIPPERS, replace(self.observation, plane_offset_m=depth))
        for values in ({"alignment_gain": 1.1}, {"tolerance_m": 0.}, {"max_camera_step_deg": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                InsertionHeightSettings(**values)
        outside = HEIGHT_Q.copy()
        outside[0] = self.planner.solver.fk.joint_limits[0, 1] + .01
        with self.assertRaisesRegex(ValueError, "공통 제한"):
            self.planner.plan(outside, GRIPPERS, self.observation)

    def test_confirmed_real_pose_is_at_goal_despite_negative_cad_body_gap(self):
        """실물 도착 자세가 CAD 간격 때문에 다시 하강 보정으로 분류되지 않는다."""
        observed = confirmed_height_observation()
        measured = self.planner.measure(CONFIRMED_Q, GRIPPERS, observed)
        self.assertAlmostEqual(measured.remaining_m, 0., places=12)
        self.assertLess(measured.lower_clearance_m, 0.)
        self.assertLess(tilt_degrees(observed.normal), 1.)
        with self.assertRaisesRegex(ValueError, "도달"):
            self.planner.plan(CONFIRMED_Q, GRIPPERS, observed)

    def test_depth_goal_is_independent_of_joint_angle_and_body_overlap(self):
        """같은 실측 거리에는 관절각이나 표시용 몸체 겹침과 무관하게 같은 높이 오차를 쓴다."""
        first = self.planner.measure(HEIGHT_Q, GRIPPERS, self.observation)
        second = self.planner.measure(NARROW_Q, GRIPPERS, self.observation)
        self.assertAlmostEqual(first.remaining_m, second.remaining_m)
        with patch("planning.insertion_height.strip_vertices", return_value=np.empty((0, 3))):
            measured = self.planner.measure(HEIGHT_Q, GRIPPERS, self.observation)
            self.assertIsNone(measured.lower_clearance_m)
            self.assertEqual(first.remaining_m, measured.remaining_m)
            self.assertGreater(self.planner.plan(HEIGHT_Q, GRIPPERS, self.observation).distance_m, 0.)

    def test_saved_reference_loads_and_invalid_values_never_fall_back(self):
        """저장한 거리 기준과 단위를 확인하고 잘못된 값에 CAD 목표를 대신 쓰지 않는다."""
        self.assertAlmostEqual(load_target_depth(), confirmed_height_observation().plane_offset_m)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            valid = json.loads(DEFAULT_DEPTH_REFERENCE_PATH.read_text())
            for value in (None, True, "171.6", -1., float("nan")):
                with self.subTest(value=value):
                    path.write_text(json.dumps({**valid, "target_depth_m": value}))
                    with self.assertRaises(ValueError):
                        load_target_depth(path)
            for field, value in (("metric", "pixel_depth"), ("reference_normal_camera", [0., 0., 0.]),
                                 ("reference_tip_camera_m", [float("nan"), 0., 0.]), ("reference_frame", "rgb_frame")):
                with self.subTest(field=field):
                    path.write_text(json.dumps({**valid, field: value}))
                    with self.assertRaises(ValueError):
                        load_height_reference(path)
            path.write_text(json.dumps({key: value for key, value in valid.items() if key != "reference_normal_camera"}))
            with self.assertRaises(ValueError):
                load_height_reference(path)

    def test_optimizer_success_alone_cannot_accept_wrong_target(self):
        """최적화 성공 표지만 있어도 실제 위치·방향 잔차가 남으면 해를 거부한다."""
        with patch("planning.vertical_motion.minimize") as optimize:
            optimize.return_value.success = True
            optimize.return_value.x = HEIGHT_Q.copy()
            with self.assertRaisesRegex(ValueError, "IK 해"):
                self.planner.plan(HEIGHT_Q, GRIPPERS, self.observation)


if __name__ == "__main__":
    unittest.main()
