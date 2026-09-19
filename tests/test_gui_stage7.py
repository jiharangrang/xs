"""가짜 모터로 뒷그리퍼 개방·당김·횡복귀의 순서와 지지 보존·중지를 검증한다."""

import asyncio
from pathlib import Path
from threading import Event
import time
import unittest
from unittest.mock import patch

import numpy as np

from gui.observed_motion import ObservedMotionSettings
from gui.continuous_motion import ContinuousJointMotion
from gui.stage7 import Stage7Session
from hardware.motor_logging import MotorLogger
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam_edge_observation import EdgeObservation
from planning.rear_pull import RearPullPlanner
import test_gui_stage1 as stage1_tests
from test_gui_stage2 import DummyReader
from test_rear_pull import AXIS, GRIPPERS, Q_DEG
from test_continuous_motion import VirtualClock


class Stage7ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """실제 서버·SDK 경로를 사용하되 카메라와 직렬 장치만 대체한다."""

    async def asyncSetUp(self):
        """몸통 토크 해제 후 양쪽 파지 상태의 가짜 장치를 준비한다."""
        await stage1_tests.Stage1ConsoleTests.asyncSetUp(self)
        angles = {**dict(zip(ARM_JOINT_NAMES, Q_DEG, strict=True)), **GRIPPERS}
        for name, angle in angles.items():
            calibration = self.controller._calibration(name)
            registers = self.chain.devices[calibration.servo_id]
            registers[56:58] = calibration.degrees_to_raw(angle).to_bytes(2, "little")
            registers[40] = int(name == "G_R")
        logger = patch("gui.observed_motion.MotorLogger", side_effect=lambda **kwargs: MotorLogger(self.pose_path.parent / "rear"))
        logger.start()
        self.addCleanup(logger.stop)
        reader = patch.object(self.console.camera.stream, "reader", side_effect=DummyReader)
        reader.start()
        self.addCleanup(reader.stop)
        self.planner = RearPullPlanner()
        self.motion_clock = VirtualClock()
        self.console.stage7 = Stage7Session(self.console, planner=self.planner, observer=self._observe,
            continuous_motion=ContinuousJointMotion(clock=self.motion_clock.clock, sleep=self.motion_clock.sleep),
            settings=ObservedMotionSettings(timeout_s=20., motion_timeout_s=1., settle_s=.001))
        self.session7 = self.console.stage7
        self.sent = []
        self.sent_options = []
        original = self.controller.move_many

        def record(targets, **kwargs):
            """실제 직렬 전송 직전 목표 순서를 기록한다."""
            self.sent.append(dict(targets))
            self.sent_options.append(dict(kwargs))
            return original(targets, **kwargs)

        recorder = patch.object(self.controller, "move_many", side_effect=record)
        recorder.start()
        self.addCleanup(recorder.stop)

    def _observe(self, reader, after_s, **kwargs):
        """저장된 빔 방향을 새 관측 시각과 함께 반환한다."""
        stamp = time.monotonic()
        return EdgeObservation(np.array([0., 0., -1.]), .173, .07, .0003, .01, stamp, stamp, 3,
                               np.array([.035, 0., .173]), AXIS, np.array([1., 0., 0.]), .0001, False)

    async def _finish(self):
        """가짜 장치의 유한한 실행이 끝날 때까지 기다린다."""
        await asyncio.wait_for(asyncio.shield(self.session7._task), timeout=18.)
        return self.session7.status()

    async def test_full_open_pull_sequence_and_api_without_closing(self):
        """기본 100 mm 실행은 현재각 유지 후 개방하고 앞 그리퍼를 건드리지 않는다."""
        front = bytes(self.chain.devices[8])
        code, _ = await stage1_tests.Stage1ConsoleTests._request(self, "/api/stage7/start")
        self.assertEqual(code, 200)
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(status["planned_progress_mm"], 100.)
        self.assertEqual(status["arrival"]["confirmation"], "joint_path_only")
        self.assertFalse(status["arrival"]["closed"])
        self.assertTrue(Path(status["path"]).exists())
        self.assertEqual(set(self.sent[0]), set(ARM_JOINT_NAMES))
        np.testing.assert_allclose([self.sent[0][name] for name in ARM_JOINT_NAMES], Q_DEG, atol=.09)
        self.assertEqual(self.sent[1], {"G_L": -120.})
        self.assertEqual(self.sent_options[1]["speed_deg_s"], 20.)
        self.assertEqual(self.sent_options[0]["speed_deg_s"], 3.)
        self.assertEqual(set(self.sent[2]), {"J1", "J7"})
        self.assertAlmostEqual(self.sent[2]["J1"] - self.sent[0]["J1"], 3., delta=.09)
        self.assertAlmostEqual(self.sent[2]["J7"] - self.sent[0]["J7"], -3., delta=.09)
        self.assertTrue(status["side_exit_completed"])
        self.assertTrue(status["side_return_completed"])
        self.assertAlmostEqual(status["planned_return_mm"], status["side_return_distance_mm"])
        self.assertEqual(status["return_reference"]["kind"], "stage7_start_lateral")
        self.assertTrue(all(set(targets) == set(ARM_JOINT_NAMES) for targets in self.sent[3:]))
        self.assertEqual(front, bytes(self.chain.devices[8]))
        self.assertAlmostEqual(status["arrival"]["positions_deg"]["G_L"], -120., delta=.09)
        code, current = await stage1_tests.Stage1ConsoleTests._request(self, "/api/stage7", "GET")
        self.assertEqual(current["arrival"], status["arrival"])

    async def test_replans_from_post_open_feedback(self):
        """개방 후 달라진 실측각에서 다시 경로를 만들어 예전 시작각으로 돌아가지 않는다."""
        original_observe = self.session7.observer
        calls = []
        original_plan = self.planner.plan

        def observe(*args, **kwargs):
            """개방 후 첫 관측 때 작은 처짐을 재현한다."""
            if self.session7._rear_open and len(calls) == 1:
                self.chain.devices[2][56:58] = (2600).to_bytes(2, "little")
            return original_observe(*args, **kwargs)

        def plan(q, *args, **kwargs):
            """각 계획이 받은 시작 자세를 기록한다."""
            calls.append(q.copy())
            return original_plan(q, *args, **kwargs)

        self.session7.observer = observe
        self.planner.plan = plan
        await self.session7.start(4.)
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(len(calls), 3)
        self.assertGreater(np.max(np.abs(calls[1] - calls[0])), .01)
        np.testing.assert_allclose(np.rad2deg(calls[2][[0, 6]] - calls[1][[0, 6]]), [3., -3.], atol=.09)

    async def test_unavailable_plan_does_not_open_or_enable_torque(self):
        """경로 실패를 개방 전에 알리고 모터 명령을 전송하지 않는다."""
        before = {i: bytes(r) for i, r in self.chain.devices.items()}
        with patch.object(self.planner, "plan", side_effect=ValueError("시험 경로 실패")):
            await self.session7.start()
            self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(self.sent, [])
        self.assertEqual(before, {i: bytes(r) for i, r in self.chain.devices.items()})

    async def test_front_support_required_without_auto_closing(self):
        """앞 그리퍼 토크가 꺼져 있으면 자동 잠금이나 개방 없이 끝난다."""
        self.chain.devices[8][40] = 0
        await self.session7.start()
        status = await self._finish()
        self.assertEqual(status["state"], "FAILED")
        self.assertIn("G_R", status["message"])
        self.assertEqual(self.sent, [])

    async def test_failed_open_does_not_start_pull(self):
        """개방 피드백이 목표에 도달하지 않으면 몸통 당김을 시작하지 않는다."""
        original = self.controller.move_many

        def blocked_open(targets, **kwargs):
            """개방 목표는 수신하지만 손가락이 움직이지 않는 상황을 만든다."""
            receipt = original(targets, **kwargs)
            if "G_L" in targets:
                self.chain.devices[0][56:58] = (2048).to_bytes(2, "little")
            return receipt

        with patch.object(self.controller, "move_many", side_effect=blocked_open):
            await self.session7.start(4.)
            self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(len(self.sent), 2)

    async def test_stop_during_planning_cancels_future_commands_and_blocks_other_stages(self):
        """계획 중 외부 정지 이후 목표가 전송되지 않으며 다른 단계와 겹치지 않는다."""
        entered, release = Event(), Event()
        original = self.planner.plan

        def delayed(*args, **kwargs):
            """명령 전송 전에 정지 요청이 들어올 시간을 만든다."""
            entered.set()
            release.wait(4.)
            return original(*args, **kwargs)

        self.planner.plan = delayed
        await self.session7.start(4.)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3.))
        try:
            for stage in (self.console.stage1, self.console.stage2, self.console.stage6):
                with self.assertRaises(MotorError):
                    await stage.start()
            with self.assertRaises(MotorError):
                await self.console.move("J1", 0., 10.)
            await self.console.stop(("J1",))
        finally:
            release.set()
        self.assertEqual((await self._finish())["state"], "STOPPED")
        self.assertEqual(self.sent, [])

    async def test_torque_off_after_open_keeps_jaw_open_without_further_pull(self):
        """개방 후 몸통 토크 해제는 이동을 중지하고 그리퍼를 다시 잠그지 않는다."""
        original = self.session7._plan_current

        async def stop_after_open(reader, filename, **kwargs):
            """두 번째 계획 직전 외부 토크 해제를 실행한다."""
            if filename == "pull_open.json":
                await self.console.set_torque(ARM_JOINT_NAMES, False)
            return await original(reader, filename, **kwargs)

        self.session7._plan_current = stop_after_open
        await self.session7.start(4.)
        self.assertEqual((await self._finish())["state"], "STOPPED")
        self.assertEqual(len(self.sent), 2)
        self.assertTrue(all(self.chain.devices[i][40] == 0 for i in range(1, 8)))
        self.assertAlmostEqual(self.controller.read("G_L").position_deg, -120., delta=.09)

    async def test_stop_during_side_exit_does_not_pull_or_close(self):
        """외측 회전 도중 중지하면 이후 당김이나 자동 잠금이 전송되지 않는다."""
        original = self.session7._wait_targets

        async def stop_side(targets):
            """두 yaw 관절 목표를 보낸 직후 사용자의 중지를 재현한다."""
            if self.session7.status()["phase"] == "side_exit":
                self.session7.request_stop()
            return await original(targets)

        self.session7._wait_targets = stop_side
        await self.session7.start(4.)
        status = await self._finish()
        self.assertEqual(status["state"], "STOPPED")
        self.assertFalse(status["side_exit_completed"])
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(set(self.sent[-1]), {"J1", "J7"})
        self.assertAlmostEqual(self.controller.read("G_L").position_deg, -120., delta=.09)

    async def test_pull_waits_for_arrival_only_once_at_the_end(self):
        """많은 연속 목표를 전송해도 당김 도중에는 도착·속도 0을 기다리지 않는다."""
        waited = []
        original = self.session7._wait_targets

        async def record_wait(targets):
            """도착 대기를 호출한 단계와 당시 목표 수를 기록한다."""
            waited.append((self.session7.status()["phase"], len(self.sent)))
            return await original(targets)

        self.session7._wait_targets = record_wait
        await self.session7.start(20.)
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual([phase for phase, _ in waited],
                         ["holding", "opening", "side_exit", "pull_arrival", "side_return_arrival"])
        self.assertGreater(waited[3][1] - waited[2][1], 10)
        self.assertEqual(status["planned_progress_mm"], 20.)

    async def test_stop_during_continuous_pull_never_sends_later_targets(self):
        """연속 당김 중 취소하면 마지막 위치를 유지하고 경로 끝이나 잠금 목표를 보내지 않는다."""
        original = self.session7._send

        async def stop_after_send(*args, **kwargs):
            """첫 연속 목표 직후 사용자의 중지를 재현한다."""
            receipt = await original(*args, **kwargs)
            if kwargs["phase"] == "pulling":
                self.session7.request_stop()
            return receipt

        self.session7._send = stop_after_send
        await self.session7.start(20.)
        status = await self._finish()
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertEqual(len(self.sent), 4)
        self.assertLess(status["planned_progress_mm"], 20.)
        self.assertAlmostEqual(self.controller.read("G_L").position_deg, -120., delta=.09)

    async def test_return_replans_from_actual_pull_arrival(self):
        """당김 후 실제 각도가 계획과 다르면 그 실측각에서 출발 횡위치로 복귀한다."""
        original_pull = self.session7._pull_continuously
        original_return = self.planner.plan_side_return
        starts = []

        async def sag_after_pull(plan):
            """당김 도착 후 J2의 실측 변화가 생기는 상황을 재현한다."""
            states = await original_pull(plan)
            calibration = self.controller._calibration("J2")
            angle = self.controller.read("J2").position_deg + .5
            self.chain.devices[2][56:58] = calibration.degrees_to_raw(angle).to_bytes(2, "little")
            starts.append(self.controller.read("J2").position_deg)
            return states

        def record_return(q, *args, **kwargs):
            """계획기에서 받은 복귀 시작각을 기록한다."""
            result = original_return(q, *args, **kwargs)
            if starts:
                self.assertAlmostEqual(np.rad2deg(q[1]), starts[-1])
            return result

        self.session7._pull_continuously = sag_after_pull
        self.planner.plan_side_return = record_return
        await self.session7.start(20.)
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertTrue(status["side_return_completed"])
        self.assertEqual(len(starts), 1)

    async def test_failed_pull_arrival_never_starts_return(self):
        """당김 도착 실패 뒤에는 빔 쪽 복귀나 잠금 명령을 보내지 않는다."""
        original = self.session7._wait_targets

        async def fail_pull(targets):
            """당김의 마지막 도착만 실패시킨다."""
            if self.session7.status()["phase"] == "pull_arrival":
                raise MotorError("시험 당김 도착 실패")
            return await original(targets)

        self.session7._wait_targets = fail_pull
        await self.session7.start(4.)
        status = await self._finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertFalse(status["side_return_completed"])
        self.assertIsNone(status["side_return_distance_mm"])
        self.assertAlmostEqual(self.controller.read("G_L").position_deg, -120., delta=.09)

    async def test_stop_during_return_never_sends_later_targets_or_closes(self):
        """횡복귀 중 취소하면 나머지 경로와 잠금 명령이 전송되지 않는다."""
        original = self.session7._send
        return_counts = []

        async def stop_return(*args, **kwargs):
            """첫 복귀 목표 직후 사용자의 중지를 재현한다."""
            receipt = await original(*args, **kwargs)
            if kwargs["phase"] == "side_return":
                return_counts.append(len(self.sent))
                self.session7.request_stop()
            return receipt

        self.session7._send = stop_return
        await self.session7.start(4.)
        status = await self._finish()
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertFalse(status["side_return_completed"])
        self.assertEqual(return_counts, [len(self.sent)])
        self.assertAlmostEqual(self.controller.read("G_L").position_deg, -120., delta=.09)
