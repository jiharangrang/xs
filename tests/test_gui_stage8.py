"""가짜 모터와 움직임에 응답하는 빔 관측으로 앞발 직진·잠금을 검증한다."""

import asyncio
from dataclasses import replace
from threading import Event
import time
import unittest
from unittest.mock import patch

import numpy as np

from gui.observed_motion import ObservedMotionSettings
from gui.stage8 import Stage8Session
from hardware.motor_logging import MotorLogger
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from planning.front_advance import FrontAdvancePlanner
from test_front_advance import compact_pose, grasp_observation, observation_from_world, sampled_preflight_case, GRIPPERS
import test_gui_stage1 as stage1_tests
from test_gui_stage2 import DummyReader


class Stage8ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """실제 콘솔·SDK를 통과시키되 장치와 카메라만 시험 대역으로 교체한다."""

    _request = stage1_tests.Stage1ConsoleTests._request

    async def asyncSetUp(self):
        """사용자가 뒷발을 잠그고 몸통 토크를 해제한 시작 상태를 준비한다."""
        await stage1_tests.Stage1ConsoleTests.asyncSetUp(self)
        angles = {**dict(zip(ARM_JOINT_NAMES, np.rad2deg(compact_pose()), strict=True)), **GRIPPERS}
        for name, angle in angles.items():
            calibration = self.controller._calibration(name)
            registers = self.chain.devices[calibration.servo_id]
            registers[56:58] = calibration.degrees_to_raw(angle).to_bytes(2, "little")
            registers[40] = int(name in ("G_L", "G_R"))
        logger = patch("gui.observed_motion.MotorLogger", side_effect=lambda **kwargs: MotorLogger(self.pose_path.parent / "front"))
        logger.start()
        self.addCleanup(logger.stop)
        reader = patch.object(self.console.camera.stream, "reader", side_effect=DummyReader)
        reader.start()
        self.addCleanup(reader.stop)
        self.planner = FrontAdvancePlanner()
        observed = grasp_observation(self.planner, self._q())
        camera = self.planner.solver.fk.depth_camera_pose(self._q())
        self.beam = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        self.calls = 0
        self.console.stage8 = Stage8Session(self.console, planner=self.planner, observer=self._observe,
            observation_interval_s=.001,
            settings=ObservedMotionSettings(timeout_s=30., motion_timeout_s=1., settle_s=.001, max_steps=200))
        self.session8 = self.console.stage8
        self.addAsyncCleanup(self.session8.stop)
        self.sent = []
        self.sent_options = []
        self.phases = []
        original = self.controller.move_many

        def record(targets, **kwargs):
            """실제 전송을 유지하면서 앞뒤 그리퍼 명령과 실행 순서를 기록한다."""
            self.sent.append(dict(targets))
            self.sent_options.append(dict(kwargs))
            return original(targets, **kwargs)

        recorder = patch.object(self.controller, "move_many", side_effect=record)
        recorder.start()
        self.addCleanup(recorder.stop)

    def _q(self):
        """가짜 엔코더의 현재 몸통 관절각을 읽는다."""
        degrees = []
        for name in ARM_JOINT_NAMES:
            calibration = self.controller._calibration(name)
            raw = int.from_bytes(self.chain.devices[calibration.servo_id][56:58], "little")
            degrees.append(calibration.raw_to_degrees(raw))
        return np.deg2rad(degrees)

    def _observe(self, reader, after_s, **kwargs):
        """현재 카메라 위치에서 고정된 빔을 새 시각에 관측한다."""
        self.calls += 1
        phase = self.session8.status()["phase"]
        if not self.phases or self.phases[-1] != phase:
            self.phases.append(phase)
        observed = observation_from_world(self.planner, self.beam, self._q())
        stamp = time.monotonic()
        return replace(observed, first_frame_s=stamp, last_frame_s=stamp,
                       single_edge=kwargs.get("reference") is not None)

    async def finish(self):
        """유한한 시험 실행의 종료 상태를 반환한다."""
        await asyncio.wait_for(asyncio.shield(self.session8._task), 25.)
        return self.session8.status()

    async def test_full_api_flow_keeps_rear_supported_and_closes_front_after_straight_advance(self):
        """옆 빼기 없이 100 mm 직진·보정 뒤 앞발만 잠그고 다음 7단계는 실행하지 않는다."""
        rear = bytes(self.chain.devices[0])
        initial = self._q().copy()
        code, _ = await self._request("/api/stage8/start")
        self.assertEqual(code, 200)
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertTrue(status["closed"])
        self.assertEqual(status["support_transfer"], "manual")
        self.assertAlmostEqual(status["progress_mm"], 100., delta=1.)
        self.assertLessEqual(abs(status["height_error_mm"]), 2.)
        self.assertEqual(status["height_tolerance_mm"], 2.)
        self.assertLessEqual(abs(status["lateral_error_mm"]), 1.)
        self.assertEqual(status["arrival"]["grip_confirmation"], "joint_angle_only")
        self.assertEqual(status["phase"], "closed_done")
        self.assertTrue(status["observation"]["single_edge"])
        self.assertEqual(self.console.stage7.status()["state"], "IDLE")
        np.testing.assert_allclose([self.sent[0][name] for name in ARM_JOINT_NAMES], np.rad2deg(initial))
        self.assertEqual(self.sent[1], {"G_R": -120.})
        self.assertEqual(self.sent[-1], {"G_R": 4.6})
        self.assertEqual(self.sent_options[1]["speed_deg_s"], 30.)
        self.assertEqual(self.sent_options[-1]["speed_deg_s"], 30.)
        self.assertEqual(self.sent_options[0]["speed_deg_s"], 3.)
        self.assertEqual(sum("G_R" in target for target in self.sent), 2)
        self.assertTrue(all("G_L" not in target for target in self.sent))
        self.assertEqual(rear, bytes(self.chain.devices[0]))
        self.assertEqual(status["motion_mode"], "straight")
        self.assertIn("advancing", self.phases)
        self.assertTrue({"side_exit", "inserting"}.isdisjoint(self.phases))
        for target in self.sent:
            if all(name in target for name in ARM_JOINT_NAMES):
                q = np.deg2rad([target[name] for name in ARM_JOINT_NAMES])
                measured = self.planner.measure(q, GRIPPERS, observation_from_world(self.planner, self.beam, q))
                self.assertLess(abs(measured.lateral_error_m), .001)
        code, current = await self._request("/api/stage8", "GET")
        self.assertEqual(current["arrival"], status["arrival"])

    async def test_manual_rear_support_is_required_without_auto_gripping(self):
        """뒷발이 열려 있거나 토크가 꺼져 있으면 앞발 개방을 포함한 명령을 보내지 않는다."""
        for torque, angle in ((False, 4.6), (True, -120.)):
            calibration = self.controller._calibration("G_L")
            self.chain.devices[0][40] = int(torque)
            self.chain.devices[0][56:58] = calibration.degrees_to_raw(angle).to_bytes(2, "little")
            await self.session8.start(20.)
            self.assertEqual((await self.finish())["state"], "FAILED")
            self.assertEqual(self.sent, [])

    async def test_residual_height_at_arrival_holds_once_then_closes(self):
        """전진 후 1.6 mm 높이 오차가 남아도 추가 상승 대신 자세 유지와 새 관측 후 잠근다."""
        original = self.session8.observer
        reached_at = None
        reached_frames = 0

        def residual_depth(*args, **kwargs):
            r"""새 관측마다 모터 보정으로 사라지지 않는 높이 잔차를 재현한다.

            $$d'=d_{goal}+0.0016,\quad p'_e=p_e-(d'-d)n$$
            """
            nonlocal reached_at, reached_frames
            observed = original(*args, **kwargs)
            if not self.session8._front_open:
                return observed
            q = self._q()
            measured = self.planner.measure(q, GRIPPERS, observed)
            # 실물 화면처럼 목표보다 1.6 mm 먼 깊이를 유지: $$d'=d_{goal}+0.0016$$
            depth = measured.goal_depth_m + .0016
            # 모서리도 같은 관측 평면 위에 유지: $$p'_e=p_e-(d'-d)n$$
            point = observed.edge_point_m - (depth - observed.plane_offset_m) * observed.normal
            if abs(measured.forward_error_m) <= .001 and abs(measured.lateral_error_m) <= .001:
                if reached_at is None:
                    reached_at = len(self.sent)
                reached_frames += 1
            return replace(observed, plane_offset_m=depth, edge_point_m=point)

        self.session8.observer = residual_depth
        await self.session8.start(50.)
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertTrue(status["closed"])
        self.assertAlmostEqual(status["height_error_mm"], 1.6)
        self.assertGreaterEqual(reached_frames, 3)
        self.assertIsNotNone(reached_at)
        self.assertEqual(len(self.sent[reached_at:]), 2)
        self.assertEqual(set(self.sent[reached_at]), set(ARM_JOINT_NAMES))
        self.assertEqual(self.sent[-1], {"G_R": 4.6})
        self.assertEqual(self.planner.height.settings.tolerance_m, .0005)

    async def test_arrival_requires_height_and_position_within_bounds(self):
        """실물 잔차는 완료로 허용하되 큰 높이·전진·횡오차와 무관하게 잠그지는 않는다."""
        states = await self.console.snapshot()
        q = self._q()
        observed = self._observe(None, 0.)
        self.planner.configure(q, GRIPPERS, observed, 50.)
        self.session8._reference_pending = False
        baseline = self.planner.measure(q, GRIPPERS, observed)
        cases = ((.0016, -.0006, -.0001, True),
                 (.003, -.0006, -.0001, False),
                 (-.003, -.0006, -.0001, False),
                 (.0016, .010, 0., False),
                 (.0016, 0., .005, False))
        for height, forward, lateral, expected in cases:
            with self.subTest(height=height, forward=forward, lateral=lateral):
                measured = replace(baseline, height_error_m=height, forward_error_m=forward,
                                   lateral_error_m=lateral)
                with patch.object(self.planner, "measure", return_value=measured):
                    arrival = await self.session8._evaluate_height(observed, states)
                self.assertEqual(arrival is not None, expected)
        self.assertEqual(self.sent, [])

    async def test_failed_preview_never_opens_front(self):
        """전진 경로를 만들지 못하면 지지 중인 앞발을 먼저 열지 않는다."""
        with patch.object(self.planner, "preview", side_effect=ValueError("시험 경로 실패")):
            await self.session8.start(20.)
            self.assertEqual((await self.finish())["state"], "FAILED")
        self.assertEqual(self.sent, [])

    async def test_recorded_sampling_failure_reaches_front_lock_with_fresh_observations(self):
        """실물 실패 자세에서 세분화된 사전 검사 후 카메라 보정과 잠금까지 이어진다."""
        q, grips, observed = sampled_preflight_case()
        angles = {**dict(zip(ARM_JOINT_NAMES, np.rad2deg(q), strict=True)), **grips}
        for name, angle in angles.items():
            calibration = self.controller._calibration(name)
            self.chain.devices[calibration.servo_id][56:58] = calibration.degrees_to_raw(angle).to_bytes(2, "little")
        camera = self.planner.solver.fk.depth_camera_pose(self._q())
        self.beam = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        await self.session8.start(100.)
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertTrue(status["closed"])
        self.assertAlmostEqual(status["progress_mm"], 100., delta=1.)
        self.assertLessEqual(abs(status["height_error_mm"]), 2.)
        self.assertLessEqual(abs(status["lateral_error_mm"]), 1.)

    async def test_opening_sag_keeps_preopen_grasp_reference(self):
        """개방 후 처짐으로 실측각이 바뀌어도 직진 기준은 파지 당시 위치를 유지한다."""
        original = self.session8.observer
        initial_point = self.planner.solver.fk.forward(self._q()).T_world_tip_R[:3, 3].copy()
        changed = False

        def sag_after_open(*args, **kwargs):
            """앞발 개방 직후 J1 실측 변화와 그에 따른 새 영상을 만든다."""
            nonlocal changed
            if self.session8._front_open and not changed:
                calibration = self.controller._calibration("J1")
                angle = self.controller.read("J1").position_deg + .5
                self.chain.devices[1][56:58] = calibration.degrees_to_raw(angle).to_bytes(2, "little")
                changed = True
            return original(*args, **kwargs)

        self.session8.observer = sag_after_open
        await self.session8.start(20.)
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertTrue(changed)
        np.testing.assert_allclose(status["advance_reference"]["point_world_m"], initial_point)

    async def test_no_waypoint_arrival_waits_during_observed_motion(self):
        """직진 중에는 도착 대기 없이 새 관측으로 목표를 갱신한다."""
        waited = []
        original = self.session8._wait_targets

        async def record_wait(targets):
            """관절 도착 대기를 요청한 구간을 기록한다."""
            waited.append(self.session8.status()["phase"])
            return await original(targets)

        self.session8._wait_targets = record_wait
        await self.session8.start(20.)
        self.assertEqual((await self.finish())["state"], "REACHED")
        self.assertEqual(waited, ["holding", "opening", "closing"])

    async def test_stop_during_forward_never_closes_or_starts_next_stage(self):
        """전진 중 사용자가 중지하면 잠금·다음 보행을 실행하지 않는다."""
        original = self.session8.observer

        def stop_forward(*args, **kwargs):
            """전진 관측 시점에서 중지를 재현한다."""
            if self.session8.status()["phase"] == "advancing":
                self.session8.request_stop()
            return original(*args, **kwargs)

        self.session8.observer = stop_forward
        await self.session8.start(20.)
        status = await self.finish()
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertNotIn({"G_R": 4.6}, self.sent)
        self.assertFalse(status["closed"])
        self.assertEqual(self.console.stage7.status()["state"], "IDLE")

    async def test_lost_beam_after_open_does_not_close(self):
        """개방 뒤 영상이 없으면 추가 전진과 자동 잠금을 하지 않는다."""
        original = self.session8.observer

        def lose_beam(*args, **kwargs):
            """앞발이 열린 뒤 관측 실패와 사용자의 중지를 재현한다."""
            if self.session8._front_open:
                self.session8.request_stop()
                raise BeamDetectionError("시험 관측 손실")
            return original(*args, **kwargs)

        self.session8.observer = lose_beam
        await self.session8.start(20.)
        status = await self.finish()
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertEqual(len(self.sent), 2)

    async def test_torque_release_and_other_stages_are_coordinated(self):
        """계획 중 다른 단계가 겹치지 않고 사용자 토크 해제 후 이동 명령이 재개되지 않는다."""
        entered, release = Event(), Event()
        original = self.planner.preview

        def delayed(*args, **kwargs):
            """개방 전 계획을 잠시 기다리게 한다."""
            entered.set()
            release.wait(4.)
            return original(*args, **kwargs)

        self.planner.preview = delayed
        await self.session8.start(20.)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3.))
        try:
            for session in (self.console.stage1, self.console.stage5, self.console.stage7):
                with self.assertRaises(MotorError):
                    await session.start()
            await self.console.set_torque(ARM_JOINT_NAMES, False)
        finally:
            release.set()
        self.assertEqual((await self.finish())["state"], "STOPPED")
        self.assertEqual(self.sent, [])

    async def test_failed_front_lock_is_not_reported_as_complete(self):
        """잠금 목표를 보내도 실제 손가락이 도착하지 않으면 완료로 표시하지 않는다."""
        original = self.controller.move_many

        def blocked_lock(targets, **kwargs):
            """잠금 시 손가락이 움직이지 않는 상황을 재현한다."""
            receipt = original(targets, **kwargs)
            if targets.get("G_R") == 4.6:
                calibration = self.controller._calibration("G_R")
                self.chain.devices[8][56:58] = calibration.degrees_to_raw(-120.).to_bytes(2, "little")
            return receipt

        with patch.object(self.controller, "move_many", side_effect=blocked_lock):
            await self.session8.start(20.)
            status = await self.finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertFalse(status["closed"])
