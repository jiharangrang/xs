"""실측 시작 자세에서 앞 팁을 고정한 개방·당김·횡복귀 경로의 기하학을 검증한다."""

import unittest

import numpy as np

from kinematics.joint_limits import load_joint_limits
from planning.rear_pull import REAR_SIDE_YAW_DEG, RearPullPlanner
from simulation.model import load_model, set_path_time


Q_DEG = [-21.09375, -49.39453125, -1.0546875, 77.6953125, -8.4375, 53.876953125, 15.64453125]
AXIS = np.array([-.27193781698, .96231482078, -.00009699204])
GRIPPERS = {"G_L": 4.39453125, "G_R": 3.779296875}


class RearPullTests(unittest.TestCase):
    """현재 자세의 오차를 강제로 제거하지 않고 상대 이동만 계산하는지 확인한다."""

    def setUp(self):
        """실물에서 저장한 각도와 기존 IK를 준비한다."""
        self.q = np.deg2rad(Q_DEG)
        self.planner = RearPullPlanner()

    def test_observed_direction_and_right_anchor_for_full_distance(self):
        """100 mm 전체 경로에서 앞 팁은 고정되고 뒷 팁은 시작 방향을 유지한다."""
        plan = self.planner.plan(self.q, GRIPPERS, AXIS)
        start = self.planner.solver.fk.forward(self.q)
        side = plan.pull.anchor.place(self.planner.solver.fk.forward(plan.side_exit.q_rad[-1]))
        np.testing.assert_array_equal(plan.motion.segments[0].q_rad[0], self.q)
        for index, q in enumerate(plan.pull.q_rad):
            placed = plan.pull.anchor.place(self.planner.solver.fk.forward(q))
            np.testing.assert_allclose(placed.T_world_tip_R, start.T_world_tip_R, atol=1e-12)
            np.testing.assert_allclose(placed.T_world_tip_L[:3, :3], side.T_world_tip_L[:3, :3], atol=.001)
            expected = side.T_world_tip_L[:3, 3] + plan.direction_world * .1 * index / 50
            np.testing.assert_allclose(placed.T_world_tip_L[:3, 3], expected, atol=.0001)
        self.assertGreater(plan.direction_world @ (start.T_world_tip_R[:3, 3] - start.T_world_tip_L[:3, 3]), 0)

    def test_axis_sign_does_not_reverse_the_pull(self):
        """검출 모서리 순서가 바뀌어도 앞 그리퍼 쪽으로 당긴다."""
        positive = self.planner.plan(self.q, GRIPPERS, AXIS, 20.)
        negative = self.planner.plan(self.q, GRIPPERS, -AXIS, 20.)
        np.testing.assert_allclose(positive.pull.q_rad, negative.pull.q_rad)

    def test_opening_never_moves_body_or_front_jaw_and_never_recloses(self):
        """개방 중 몸통과 앞턱을 유지하고 당김 끝에도 뒷턱이 열린 채 남는다."""
        plan = self.planner.plan(self.q, GRIPPERS, AXIS, 20.)
        opening, side, pull, returning = plan.motion.segments
        np.testing.assert_allclose(opening.q_rad, [self.q, self.q])
        np.testing.assert_allclose(np.rad2deg(opening.gripper_q_rad[:, 1]), GRIPPERS["G_R"])
        np.testing.assert_allclose(np.rad2deg(side.gripper_q_rad[:, 0]), -120.)
        np.testing.assert_allclose(np.rad2deg(pull.gripper_q_rad[:, 0]), -120.)
        np.testing.assert_allclose(np.rad2deg(pull.gripper_q_rad[:, 1]), GRIPPERS["G_R"])
        np.testing.assert_allclose(np.rad2deg(returning.gripper_q_rad),
                                   np.tile([-120., GRIPPERS["G_R"]], (len(returning.q_rad), 1)))
        plan.motion.validate_limits()

    def test_both_gripper_limits_allow_122_degree_opening(self):
        """앞뒤 공통 설정에 사용자가 지정한 개방 한도가 남는다."""
        limits = load_joint_limits(("G_L", "G_R"))
        for name in limits:
            self.assertAlmostEqual(np.rad2deg(limits[name][0]), -122.)
            self.assertAlmostEqual(np.rad2deg(limits[name][1]), 5.)

    def test_fixed_rotation_moves_rear_fixed_jaw_outward_in_mujoco(self):
        """MuJoCo의 뒷 고정턱 메시가 바깥으로 이동하고 반대 회전은 안쪽임을 검증한다."""
        plan = self.planner.plan(self.q, GRIPPERS, AXIS)
        opening, side = plan.motion.segments[:2]
        np.testing.assert_allclose(np.rad2deg(side.q_rad[-1] - self.q), [3., 0., 0., 0., 0., 0., -3.])
        model, data = load_model(self.q)
        set_path_time(model, data, plan.motion, opening.duration_s)
        geom = model.geom("gripper_L_geom_0").id
        mesh = model.geom_dataid[geom]
        begin = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[begin:begin + model.mesh_vertnum[mesh]]
        fixed_before = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        body = data.body("gripper_L")
        local = (fixed_before - body.xpos) @ body.xmat.reshape(3, 3)
        lip = local[:, 2] >= model.site("tip_L").pos[2] - 1e-6
        self.assertGreater(int(np.count_nonzero(lip)), 4)
        self.assertGreater(float(np.median(local[lip, 1])), 0.)
        outward = body.xmat.reshape(3, 3)[:, 1].copy()
        front = data.site("tip_R").xpos.copy()
        previous = -1e-12
        for fraction in np.linspace(0., 1., 9):
            set_path_time(model, data, plan.motion, opening.duration_s + fraction * side.duration_s)
            fixed_now = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            moved = float(np.min((fixed_now[lip] - fixed_before[lip]) @ outward))
            self.assertGreaterEqual(moved + 1e-12, previous)
            np.testing.assert_allclose(data.site("tip_R").xpos, front, atol=1e-12)
            previous = moved
        self.assertGreater(previous, .010)
        reversed_q = self.q.copy()
        reversed_q[[0, 6]] += np.deg2rad([-REAR_SIDE_YAW_DEG, REAR_SIDE_YAW_DEG])
        reversed_pose = side.anchor.place(self.planner.solver.fk.forward(reversed_q))
        before = side.anchor.place(self.planner.solver.fk.forward(self.q))
        self.assertLess((reversed_pose.T_world_tip_L[:3, 3] - before.T_world_tip_L[:3, 3]) @ outward, 0.)

    def test_replan_after_side_exit_keeps_offset_and_anchor_without_second_rotation(self):
        """옆으로 빠진 뒤에는 같은 앞 팁 좌표계에서 당기며 외측 회전을 반복하지 않는다."""
        full = self.planner.plan(self.q, GRIPPERS, AXIS)
        q_side = full.side_exit.q_rad[-1]
        again = self.planner.plan(q_side, {**GRIPPERS, "G_L": -120.}, AXIS, 20.,
                                  include_side_exit=False, anchor=full.pull.anchor, return_reference=full.return_reference)
        self.assertIsNone(again.side_exit)
        np.testing.assert_array_equal(again.pull.q_rad[0], q_side)
        np.testing.assert_array_equal(again.pull.anchor.T_world_fixed_tip, full.pull.anchor.T_world_fixed_tip)
        self.assertIs(again.return_reference, full.return_reference)
        after_side = full.pull.anchor.place(self.planner.solver.fk.forward(q_side))
        after_pull = again.pull.anchor.place(self.planner.solver.fk.forward(again.pull.q_rad[-1]))
        expected = after_side.T_world_tip_L[:3, 3] + .020 * again.direction_world
        np.testing.assert_allclose(after_pull.T_world_tip_L[:3, 3], expected, atol=.0001)

    def test_side_return_reaches_start_lateral_coordinate_in_mujoco(self):
        """당김 길이가 달라도 앞 팁·높이·방향을 유지하며 출발 횡위치까지 복귀한다."""
        for distance in (20., 100.):
            with self.subTest(distance=distance):
                plan = self.planner.plan(self.q, GRIPPERS, AXIS, distance)
                segment = plan.side_return
                reference = plan.return_reference
                model, data = load_model(self.q)
                before_s = plan.motion.duration_s - segment.duration_s
                set_path_time(model, data, plan.motion, before_s)
                start = data.site("tip_L").xpos.copy()
                rotation = data.site("tip_L").xmat.copy()
                front = data.site("tip_R").xpos.copy()
                geom = model.geom("gripper_L_geom_0").id
                fixed_jaw = data.geom_xpos[geom].copy()
                previous = float((start - reference.point_world_m) @ reference.outward_world)
                self.assertGreater(previous, .010)
                for fraction in np.linspace(0., 1., 17):
                    set_path_time(model, data, plan.motion, before_s + fraction * segment.duration_s)
                    point = data.site("tip_L").xpos.copy()
                    remaining = float((point - reference.point_world_m) @ reference.outward_world)
                    self.assertLessEqual(remaining, previous + .0001)
                    self.assertGreaterEqual(remaining, -.0001)
                    delta = point - start
                    orthogonal = delta - float(delta @ reference.outward_world) * reference.outward_world
                    np.testing.assert_allclose(orthogonal, 0., atol=.0001)
                    np.testing.assert_allclose(data.site("tip_R").xpos, front, atol=1e-12)
                    np.testing.assert_allclose(data.site("tip_L").xmat, rotation, atol=.001)
                    previous = remaining
                self.assertAlmostEqual(previous, 0., delta=.0001)
                self.assertLess(float((data.geom_xpos[geom] - fixed_jaw) @ reference.outward_world), -.010)

    def test_post_exit_plan_requires_original_return_reference(self):
        """옆으로 빠진 상태를 새 복귀 원점으로 잘못 쓰지 못하게 한다."""
        with self.assertRaisesRegex(ValueError, "복귀 기준"):
            self.planner.plan(self.q, GRIPPERS, AXIS, include_side_exit=False)

    def test_invalid_requests_do_not_create_paths(self):
        """잘못된 거리와 검출 방향으로 경로를 만들지 않는다."""
        for distance in (0., -1., 101., np.nan, np.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.planner.plan(self.q, GRIPPERS, AXIS, distance)
        for axis in ([0., 0., 0.], [1., 0.], [np.nan, 0., 0.]):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                self.planner.plan(self.q, GRIPPERS, axis)
