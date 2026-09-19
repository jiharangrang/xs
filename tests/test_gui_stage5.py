"""가짜 모터로 높이·정면 동시 보정과 이동 중 재관측·수동 정지를 검증한다."""

import asyncio
from dataclasses import replace
from threading import Event
import time
import unittest

import numpy as np

from gui.observed_motion import ObservedMotionSettings
from gui.stage5 import Stage5Session
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from planning.beam_alignment import tilt_degrees
from planning.insertion_height import InsertionHeightPlanner
from test_insertion_height import (HEIGHT_Q, TILTED_Q, CONFIRMED_Q, confirmed_height_observation, height_observation,
                                   observation_at, tilted_height_observation, CONTACT_Q, contact_height_observation, flat_goal_depth)
import test_gui_stage4 as stage4_tests


class Stage5ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """실제 SDK 패킷과 제어기를 사용하고 관측·직렬 장치만 시험 대역으로 바꾼다."""

    _q = stage4_tests.Stage4ConsoleTests._q
    _request = stage4_tests.Stage4ConsoleTests._request
    _move_count = stage4_tests.Stage4ConsoleTests._move_count
    _observe = stage4_tests.Stage4ConsoleTests._observe

    async def asyncSetUp(self):
        """외측 이동이 끝난 자세와 카메라 관측 대역을 준비한다."""
        await stage4_tests.Stage4ConsoleTests.asyncSetUp(self)
        for name, angle in zip(ARM_JOINT_NAMES, np.rad2deg(HEIGHT_Q), strict=True):
            calibration = self.controller._calibration(name)
            raw = calibration.degrees_to_raw(float(angle))
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        planner = InsertionHeightPlanner()
        self.console.stage5 = Stage5Session(self.console, planner=planner, observer=self._observe,
                                            settings=ObservedMotionSettings(settle_s=.001, timeout_s=45., max_steps=40),
                                            observation_interval_s=.001)
        self.session5 = self.console.stage5
        self.set_beam(.22)
        self.addAsyncCleanup(self.session5.stop)

    def set_beam(self, depth):
        """원하는 초기 깊이의 빔을 현재 카메라에서 공간 기준으로 고정한다."""
        observed = height_observation(self.session5.planner, self._q(), depth)
        camera = self.fk.depth_camera_pose(self._q())
        self.reference = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        return observed

    async def finish(self):
        """전체 실행이 제한된 시간 안에 완료되도록 기다린다."""
        await asyncio.wait_for(asyncio.shield(self.session5._task), 35.)
        return self.session5.status()

    async def test_height_and_frontal_goal_reached_with_grippers_unchanged(self):
        """높이·정면을 새 관측으로 확인하고 마지막 구간은 낮은 속도로 움직인다."""
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        speeds = []
        original = self.console.move_pose

        async def record(*args, **kwargs):
            """실제 전송 경로를 유지하면서 요청 속도를 기록한다."""
            speeds.append(kwargs["speed_deg_s"])
            return await original(*args, **kwargs)

        self.console.move_pose = record
        code, _ = await self._request("/api/stage5/start")
        self.assertEqual(code, 200)
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertLessEqual(abs(status["remaining_mm"]), .5)
        self.assertLessEqual(status["tilt_deg"], 1.)
        self.assertTrue(status["holding_goal"])
        self.assertIn(3., speeds)
        self.assertIn(1., speeds)
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})
        code, current = await self._request("/api/stage5", "GET")
        self.assertEqual(current["arrival"], status["arrival"])

    async def test_confirmed_real_depth_completes_without_another_height_move(self):
        """실물에서 확인한 현재 깊이와 자세를 재현하면 추가 상승·하강 없이 완료한다."""
        for name, angle in zip(ARM_JOINT_NAMES, np.rad2deg(CONFIRMED_Q), strict=True):
            calibration = self.controller._calibration(name)
            raw = calibration.degrees_to_raw(float(angle))
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        camera = self.fk.depth_camera_pose(self._q())
        observed = confirmed_height_observation()
        self.reference = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        before = self._q().copy()
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(status["step"], 1)
        self.assertEqual(status["motion_kind"], "hold")
        self.assertEqual(status["commanded_lift_mm"], 0.)
        self.assertAlmostEqual(status["depth_mm"], status["goal_depth_mm"])
        self.assertLess(status["lower_clearance_mm"], 0.)
        np.testing.assert_array_equal(self._q(), before)

    async def test_contact_height_finishes_despite_frontal_error(self):
        """접촉 화면의 기울기가 허용각 밖이어도 높이가 맞으면 유지 명령만 보내고 완료한다."""
        for name, angle in zip(ARM_JOINT_NAMES, np.rad2deg(CONTACT_Q), strict=True):
            calibration = self.controller._calibration(name)
            raw = calibration.degrees_to_raw(float(angle))
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        camera = self.fk.depth_camera_pose(self._q())
        observed = contact_height_observation()
        self.reference = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        before = self._q().copy()
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertGreater(status["tilt_deg"], 1.)
        self.assertEqual(status["commanded_lift_mm"], 0.)
        self.assertEqual(status["step"], 1)
        self.assertEqual(status["motion_kind"], "hold")
        self.assertAlmostEqual(status["remaining_mm"], -.342318674, places=6)
        self.assertGreater(status["goal_depth_mm"], 173.)
        np.testing.assert_array_equal(self._q(), before)

    async def test_moving_feedback_and_old_time_count_limits_do_not_block(self):
        """속도·도착 오차가 남아도 전체 시간·횟수·안정 대기로 중단하지 않는다."""
        original = self.console.snapshot

        async def still_moving():
            """몸통 엔코더는 유지하고 모터 도착 판정만 계속 미완료로 만든다."""
            states = await original()
            for state in states:
                if state["name"] in ARM_JOINT_NAMES:
                    state.update(speed_deg_s=2., arrived_now=False, motion_status="timeout", error_deg=2.)
            return states

        self.console.snapshot = still_moving
        self.session5.settings = replace(self.session5.settings, timeout_s=.01, motion_timeout_s=.01, max_steps=1)
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertGreater(status["step"], 1)
        self.assertLessEqual(abs(status["remaining_mm"]), .5)

    async def test_real_tilt_and_height_are_corrected_in_first_same_move(self):
        r"""실물 실패 관측에서 정면 복구를 따로 기다리지 않고 첫 명령부터 상승한다.

        $$q_{rad}=q_{deg}\pi/180$$
        """
        for name, angle in zip(ARM_JOINT_NAMES, np.rad2deg(TILTED_Q), strict=True):
            calibration = self.controller._calibration(name)
            raw = calibration.degrees_to_raw(float(angle))
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        camera = self.fk.depth_camera_pose(self._q())
        observed = tilted_height_observation()
        self.reference = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        motions = []
        original = self.session5._update

        def record(state, message, **values):
            """첫 관절 목표가 상승과 회전을 함께 반영했는지 확인할 상태를 보관한다."""
            original(state, message, **values)
            if state == "MOVING":
                motions.append(dict(values))

        self.session5._update = record
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(motions[0]["motion_kind"], "lift")
        self.assertGreater(motions[0]["commanded_lift_mm"], 0.)
        # 전송된 첫 목표를 기구학 입력으로 변환: $$q_{rad}=q_{deg}\pi/180$$
        first_q = np.deg2rad([motions[0]["targets_deg"][name] for name in ARM_JOINT_NAMES])
        shifted = observation_at(self.session5.planner, observed, TILTED_Q, first_q)
        self.assertLess(tilt_degrees(shifted.normal), 1.5)
        self.assertLessEqual(status["tilt_deg"], 1.)
        self.assertLessEqual(abs(status["remaining_mm"]), .5)

    async def test_overshot_height_can_lower_and_reach(self):
        """지나친 높이는 작은 하강 보정으로 되돌아와 완료한다."""
        self.set_beam(self.session5.height_settings.target_depth_m - .0014)
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertLess(status["commanded_lift_mm"], 0.)

    async def test_small_side_clearance_and_empty_window_are_diagnostics(self):
        """여유가 작은 관측도 목표 높이 추종을 막거나 삽입 성공으로 해석하지 않는다."""
        settings = replace(self.session5.height_settings, flange_thickness_m=.012)
        self.session5.height_settings = settings
        self.session5.planner = InsertionHeightPlanner(settings=settings)
        initial = self.set_beam(.186)
        shifted = replace(initial, edge_point_m=initial.edge_point_m + .008 * initial.outward)
        camera = self.fk.depth_camera_pose(self._q())
        self.reference = shifted.reference().transformed(camera[:3, :3], camera[:3, 3])
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertLess(status["side_clearance_mm"], 3.)
        self.assertLess(status["upper_clearance_mm"] + status["lower_clearance_mm"], 0.)
        self.assertIn("목표 높이를", status["message"])

    async def test_missing_depth_holds_then_resumes_with_stage4_reference(self):
        """4단계 모서리를 넘겨받고 일시적인 영상 누락 이후 새 깊이로 계속한다."""
        initial = self.set_beam(.177)
        self.console.stage4._status.update(state="REACHED", observation=initial.as_dict(),
                                           arrival={"positions_deg": dict(zip(ARM_JOINT_NAMES, np.rad2deg(self._q()), strict=True))})
        original, missed = self._observe, []
        count = self._move_count()

        def interrupted(*args, **kwargs):
            """최초 관측만 누락시키고 모서리 참고값과 전송 억제를 검사한다."""
            self.assertIsNotNone(kwargs["reference"])
            if not missed:
                missed.append(True)
                raise BeamDetectionError("깊이 누락")
            if self.calls == 0:
                self.assertEqual(self._move_count(), count)
            return original(*args, **kwargs)

        self.session5.observer = interrupted
        await self.session5.start()
        self.assertEqual((await self.finish())["state"], "REACHED")

    async def test_stale_or_missing_depth_waits_without_failed_timeout(self):
        """깊이가 없거나 과거 영상이면 새 명령 없이 기다리고 수동 정지를 받는다."""
        original = self._observe

        def stale(*args, **kwargs):
            """현재 관측에 과거 시각을 부여한다."""
            return replace(original(*args, **kwargs), first_frame_s=0., last_frame_s=0.)

        def missing(*args, **kwargs):
            """유효한 깊이가 전혀 없는 상황을 만든다."""
            raise BeamDetectionError("깊이 누락")

        self.session5.settings = replace(self.session5.settings, timeout_s=.01)
        self.session5.observation_interval_s = .01
        for observer in (stale, missing):
            with self.subTest(observer=observer.__name__):
                count = self._move_count()
                self.session5.observer = observer
                await self.session5.start()
                await asyncio.sleep(.08)
                self.assertTrue(self.session5.active)
                self.assertEqual(self._move_count(), count)
                self.assertEqual((await self.session5.stop())["state"], "STOPPED")

    async def test_no_height_progress_does_not_trigger_repeat_limit(self):
        """높이 변화가 느려도 반복 횟수로 실패시키지 않고 다음 관측을 처리한다."""
        initial = self._observe(None, 0., cancelled=lambda: False, outward_hint=None, reference=None)
        sent = asyncio.Event()
        original = self.console.move_pose
        baseline = self._move_count()

        def unchanged(*args, **kwargs):
            """처짐 등으로 실제 관측 높이가 변하지 않는 새 영상을 만든다."""
            stamp = time.monotonic()
            return replace(initial, first_frame_s=stamp, last_frame_s=stamp)

        async def record(*args, **kwargs):
            """기존 무진행 한도보다 여러 번 명령을 받은 뒤 시험을 중지한다."""
            receipt = await original(*args, **kwargs)
            if self._move_count() - baseline >= 5:
                self.session5.request_stop()
                sent.set()
            return receipt

        self.session5.observer = unchanged
        self.console.move_pose = record
        self.session5.settings = replace(self.session5.settings, max_steps=1)
        await self.session5.start()
        await asyncio.wait_for(sent.wait(), 5.)
        status = await self.finish()
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertIn("사용자", status["message"])
        self.assertGreaterEqual(status["step"], 5)

    async def test_reached_camera_holds_current_position_before_fresh_confirmation(self):
        """이전 모터 목표가 남아 있어도 관측 도착 시 현재각 유지로 바꾼 뒤 완료한다."""
        self.set_beam(flat_goal_depth(self.session5.planner, self._q()))
        calibration = self.controller._calibration("J2")
        device = self.chain.devices[2]
        raw = int.from_bytes(device[56:58], "little")
        device[42:44] = (raw + 20).to_bytes(2, "little")
        original, held = self.console.move_pose, []

        async def record(targets, **kwargs):
            """도착 유지 목표가 새 관측의 실측각인지 검사한다."""
            self.assertAlmostEqual(targets["J2"], calibration.raw_to_degrees(raw))
            self.assertNotEqual(self.session5.status()["state"], "REACHED")
            held.append(self.calls)
            return await original(targets, **kwargs)

        self.console.move_pose = record
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(len(held), 1)
        self.assertGreaterEqual(self.calls - held[0], 2)
        self.assertEqual(device[42:44], device[56:58])

    async def test_drift_after_hold_restarts_height_correction(self):
        """도착 유지 명령 후 높이가 달라지면 완료하지 않고 다시 보정한다."""
        self.set_beam(flat_goal_depth(self.session5.planner, self._q()))
        original, moves = self.console.move_pose, []

        async def disturbed(targets, **kwargs):
            """첫 유지 명령 직후 빔과의 거리가 달라진 상황을 만든다."""
            receipt = await original(targets, **kwargs)
            moves.append(receipt)
            if len(moves) == 1:
                self.set_beam(.177)
            return receipt

        self.console.move_pose = disturbed
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertGreater(status["commanded_lift_mm"], 1.)
        self.assertGreater(len(moves), 2)

    async def test_stop_during_planning_excludes_other_stages_and_cancels_send(self):
        """계획 중 다른 단계와 수동 이동을 차단하고 정지 뒤 새 패킷을 보내지 않는다."""
        entered, release = Event(), Event()
        original = self.session5.planner.plan

        def delayed(*args):
            """계산 중간에 정지 요청을 넣을 시간을 확보한다."""
            entered.set()
            release.wait(3.)
            return original(*args)

        self.session5.planner.plan = delayed
        count = self._move_count()
        await self.session5.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 3.))
        for session in (self.console.stage1, self.console.stage2, self.console.stage3, self.console.stage4):
            with self.assertRaises(MotorError):
                await session.start()
        with self.assertRaises(MotorError):
            await self.console.move_pose(dict(zip(ARM_JOINT_NAMES, np.rad2deg(self._q()), strict=True)))
        self.session5.request_stop()
        release.set()
        self.assertEqual((await self.finish())["state"], "STOPPED")
        self.assertEqual(self._move_count(), count)

    async def test_user_stop_during_observation_interval_keeps_current_pose(self):
        """다음 관측 주기를 기다리는 중 사용자 정지는 즉시 현재각 유지로 처리한다."""
        sent = asyncio.Event()
        original = self.console.move_pose
        self.session5.observation_interval_s = 1.

        async def moving(*args, **kwargs):
            """첫 목표와 다른 실제각을 남겨 정지 명령의 현재각 유지를 검증한다."""
            receipt = await original(*args, **kwargs)
            device = self.chain.devices[2]
            raw = int.from_bytes(device[42:44], "little")
            device[56:58] = (raw + 19).to_bytes(2, "little")
            sent.set()
            return receipt

        self.console.move_pose = moving
        count = self._move_count()
        await self.session5.start()
        await asyncio.wait_for(sent.wait(), 3.)
        status = await asyncio.wait_for(self.session5.stop(), .5)
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertEqual(self._move_count() - count, 1)
        self.assertEqual(self.chain.devices[2][42:44], self.chain.devices[2][56:58])

    async def test_invalid_feedback_and_unreachable_goal_do_not_send(self):
        """장치 피드백이 없거나 유효한 IK 목표를 못 구하면 명령을 만들지 않는다."""
        original = self.console.snapshot
        count = self._move_count()

        async def invalid():
            """실제 통신 경로에서 얻은 상태 중 하나를 장치 오류로 바꾼다."""
            states = await original()
            next(state for state in states if state["name"] == "J2")["torque"] = False
            return states

        self.console.snapshot = invalid
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertIn("J2", status["message"])
        self.assertEqual(self._move_count(), count)
        self.console.snapshot = original

        def unreachable(*args):
            """공통 관절 범위 내에서 해를 찾지 못한 경우를 만든다."""
            raise ValueError("IK 해를 찾지 못했습니다.")

        self.session5.planner.plan = unreachable
        await self.session5.start()
        status = await self.finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertIn("IK 해", status["message"])
        self.assertEqual(self._move_count(), count)


if __name__ == "__main__":
    unittest.main()
