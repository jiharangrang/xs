"""가짜 모터와 저장된 실물 관측으로 6단계의 완료·중지·모서리 추적을 검증한다."""

import asyncio
from dataclasses import replace
from threading import Event
import unittest

import numpy as np

from gui.observed_motion import ObservedMotionSettings
from gui.stage6 import Stage6Session
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from planning.beam_insertion import BeamInsertionPlanner
from planning.insertion_reference import save_lateral_reference
from test_beam_insertion import return_reference
from test_insertion_height import CONFIRMED_Q, confirmed_height_observation, CONTACT_Q, contact_height_observation
import test_gui_stage4 as stage4_tests


class Stage6ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """실제 SDK·콘솔을 사용하고 직렬 장치와 카메라 관측만 시험 대역으로 바꾼다."""

    _q = stage4_tests.Stage4ConsoleTests._q
    _request = stage4_tests.Stage4ConsoleTests._request
    _move_count = stage4_tests.Stage4ConsoleTests._move_count
    _observe = stage4_tests.Stage4ConsoleTests._observe

    async def asyncSetUp(self):
        """사용자가 도착으로 확인한 자세에서 새 6단계를 준비한다."""
        await stage4_tests.Stage4ConsoleTests.asyncSetUp(self)
        for name, angle in zip(ARM_JOINT_NAMES, np.rad2deg(CONFIRMED_Q), strict=True):
            calibration = self.controller._calibration(name)
            raw = calibration.degrees_to_raw(float(angle))
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        self.console.stage6 = Stage6Session(self.console, planner=BeamInsertionPlanner(), observer=self._observe,
                                            settings=ObservedMotionSettings(timeout_s=.01, max_steps=1),
                                            observation_interval_s=.001)
        self.session6 = self.console.stage6
        self.addAsyncCleanup(self.session6.stop)
        self.set_beam(confirmed_height_observation())
        self.set_return_goal(confirmed_height_observation())

    def set_return_goal(self, observation, distance=.024):
        """현재 위치에서 알려진 거리만큼 안쪽인 4단계 출발 기준을 저장한다."""
        reference = return_reference(self.session6.planner, self._q(), observation, distance)
        save_lateral_reference(reference, self.console.stage4.reference_path)

    def set_beam(self, observation):
        """지정한 관측을 현재 카메라에서 공간에 고정된 빔으로 저장한다."""
        camera = self.fk.depth_camera_pose(self._q())
        self.reference = observation.reference().transformed(camera[:3, :3], camera[:3, 3])

    async def finish(self):
        """시험 중 실행 작업이 끝나면 최종 상태를 반환한다."""
        await asyncio.wait_for(asyncio.shield(self.session6._task), 12.)
        return self.session6.status()

    async def test_api_returns_to_stage4_lateral_position_preserving_height_and_grippers(self):
        """6단계 API가 출발 횡위치까지 복귀하고 그리퍼 개폐 명령을 보내지 않는다."""
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        code, _ = await self._request("/api/stage6/start")
        self.assertEqual(code, 200)
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertLessEqual(abs(status["remaining_mm"]), 1.)
        self.assertLessEqual(abs(status["height_error_mm"]), .5)
        self.assertLessEqual(status["tilt_deg"], 1.)
        self.assertGreater(status["overlap_mm"], 10.)
        self.assertAlmostEqual(status["commanded_insert_mm"], 24., delta=2.)
        self.assertEqual(status["arrival"]["confirmation"], "stage4_lateral_and_camera_height")
        self.assertEqual(status["arrival"]["source_stage4_run_id"], "stage4-test-run")
        self.assertFalse(status["arrival"]["gripper_commanded"])
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})
        self.assertTrue(status["holding_goal"])
        self.assertGreaterEqual(self.calls, status["step"] + 2)
        code, current = await self._request("/api/stage6", "GET")
        self.assertEqual(code, 200)
        self.assertEqual(current["arrival"], status["arrival"])

    async def test_frontal_error_does_not_block_height_ready_insertion_or_completion(self):
        """정면 오차를 완료 조건으로 삼지 않고 높이와 횡위치가 맞으면 삽입을 마친다."""
        for name, angle in zip(ARM_JOINT_NAMES, np.rad2deg(CONTACT_Q), strict=True):
            calibration = self.controller._calibration(name)
            raw = calibration.degrees_to_raw(float(angle))
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        observed = contact_height_observation()
        before = self._q().copy()
        self.set_beam(observed)
        self.set_return_goal(observed, distance=0.)
        await self.session6.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertGreater(status["tilt_deg"], 1.)
        self.assertEqual(status["phase"], "inserting")
        self.assertEqual(status["commanded_insert_mm"], 0.)
        self.assertEqual(status["commanded_lift_mm"], 0.)
        self.assertEqual(status["step"], 1)
        self.assertEqual(status["motion_kind"], "hold")
        np.testing.assert_array_equal(self._q(), before)

    async def test_restart_uses_saved_stage4_reference_without_live_stage_history(self):
        """서버 재시작 후에는 저장한 출발 횡위치와 새 카메라 관측으로 복귀한다."""
        self.console.stage4._status.update(state="IDLE", arrival=None, observation=None)
        self.console.stage5._status.update(state="IDLE", arrival=None, observation=None)
        await self.session6.start()
        self.assertEqual((await self.finish())["state"], "REACHED")

    async def test_missing_stage4_reference_does_not_send_rgb_center_goal(self):
        """출발 기준이 없으면 RGB 중앙으로 대신 삽입하지 않고 필요한 기록을 안내한다."""
        self.console.stage4.reference_path.unlink()
        count = self._move_count()
        await self.session6.start()
        status = await self.finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertIn("4단계", status["message"])
        self.assertEqual(self._move_count(), count)

    async def test_repeated_run_keeps_same_goal_without_replaying_exit_distance(self):
        """도착 후 다시 실행해도 전체 횡이동을 반복하지 않고 현재 위치를 유지한다."""
        await self.session6.start()
        first = await self.finish()
        self.assertEqual(first["state"], "REACHED", first)
        before = self._q().copy()
        await self.session6.start()
        repeated = await self.finish()
        self.assertEqual(repeated["state"], "REACHED", repeated)
        self.assertEqual(repeated["commanded_insert_mm"], 0.)
        self.assertEqual(repeated["motion_kind"], "hold")
        self.assertEqual(repeated["lateral_reference"], first["lateral_reference"])
        np.testing.assert_array_equal(self._q(), before)

    async def test_partial_retry_only_moves_remaining_return_distance(self):
        """중간 정지 후 재실행해도 출발 기준을 새로 잡지 않고 남은 거리만 복귀한다."""
        sent = asyncio.Event()
        original = self.console.move_pose
        self.session6.observation_interval_s = 1.

        async def record(*args, **kwargs):
            """첫 삽입 목표 전송 이후에 정지를 요청할 수 있게 알린다."""
            result = await original(*args, **kwargs)
            sent.set()
            return result

        self.console.move_pose = record
        await self.session6.start()
        await asyncio.wait_for(sent.wait(), 3.)
        await self.session6.stop()
        partial = self.session6.status()
        self.assertEqual(partial["state"], "STOPPED")
        self.assertAlmostEqual(partial["commanded_insert_mm"], 3.)
        self.session6.observation_interval_s = .001
        await self.session6.start()
        completed = await self.finish()
        self.assertEqual(completed["state"], "REACHED", completed)
        self.assertAlmostEqual(completed["commanded_insert_mm"], 21., delta=2.)
        self.assertEqual(completed["lateral_reference"], partial["lateral_reference"])

    async def test_new_live_stage4_record_takes_priority_over_saved_reference(self):
        """같은 서버에서 새 4단계가 완료되면 오래된 저장 파일보다 새 출발 위치를 쓴다."""
        reference = return_reference(self.session6.planner, self._q(), confirmed_height_observation(), .006)
        self.console.stage4._status.update(
            state="REACHED", run_id="new-run", observation=confirmed_height_observation().as_dict(),
            arrival={"start_tip_world_m": reference.point_world_m,
                     "positions_deg": dict(zip(ARM_JOINT_NAMES, np.rad2deg(self._q()), strict=True))})
        await self.session6.start()
        completed = await self.finish()
        self.assertEqual(completed["state"], "REACHED", completed)
        self.assertEqual(completed["arrival"]["source_stage4_run_id"], "new-run")
        self.assertAlmostEqual(completed["commanded_insert_mm"], 6., delta=2.)

    async def test_stage5_reference_tracks_single_edge_and_recovers_detection_loss(self):
        """5단계에서 확인한 폭을 이어받고 잠깐 모서리를 놓친 뒤 같은 빔을 계속 추적한다."""
        observed = confirmed_height_observation()
        self.console.stage5._status.update(state="REACHED", observation=observed.as_dict(),
                                           arrival={"positions_deg": dict(zip(ARM_JOINT_NAMES, np.rad2deg(self._q()), strict=True))})
        original, missed = self._observe, []
        count = self._move_count()

        def interrupted(*args, **kwargs):
            """첫 관측을 누락시키고 이후 한쪽 모서리 추적으로 복귀한다."""
            self.assertIsNotNone(kwargs["reference"])
            if not missed:
                missed.append(True)
                raise BeamDetectionError("일시적인 모서리 누락")
            if self.calls == 0:
                self.assertEqual(self._move_count(), count)
            return replace(original(*args, **kwargs), single_edge=True)

        self.session6.observer = interrupted
        await self.session6.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertTrue(status["observation"]["single_edge"])

    async def test_initial_wrong_height_is_corrected_before_first_lateral_move(self):
        """높이 준비가 안 된 시작 자세는 실패시키지 않고 높이를 맞춘 뒤 삽입한다."""
        observed = confirmed_height_observation()
        self.set_beam(replace(observed, plane_offset_m=observed.plane_offset_m + .003,
                              edge_point_m=observed.edge_point_m - .003 * observed.normal))
        motions = []
        original = self.session6._update

        def record(state, message, **values):
            """실제 전송이 기록된 시점의 삽입 준비 상태를 보관한다."""
            original(state, message, **values)
            if state == "MOVING":
                motions.append(self.session6.status())

        self.session6._update = record
        await self.session6.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(motions[0]["phase"], "aligning")
        self.assertEqual(motions[0]["commanded_insert_mm"], 0.)
        first_insert = next(value for value in motions if value["commanded_insert_mm"] > 0.)
        self.assertLessEqual(abs(first_insert["height_error_mm"]), .5)

    async def test_moving_feedback_does_not_wait_for_joint_arrival(self):
        """카메라 오차가 줄면 관절 안정 신호나 기존 시간·횟수 한도로 삽입을 막지 않는다."""
        original = self.console.snapshot

        async def moving():
            """실측각은 유지하고 속도와 도착 판정만 미완료로 제공한다."""
            states = await original()
            for state in states:
                if state["name"] in ARM_JOINT_NAMES:
                    state.update(speed_deg_s=1., arrived_now=False, motion_status="timeout")
            return states

        self.console.snapshot = moving
        await self.session6.start()
        status = await self.finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertGreater(status["step"], 1)

    async def test_missing_depth_never_replays_old_return_displacement(self):
        """새 관측을 못 얻으면 이전 횡이동을 반전해 보내거나 완료를 추정하지 않는다."""
        def missing(*args, **kwargs):
            """새 빔 깊이를 확인할 수 없는 상황을 만든다."""
            raise BeamDetectionError("모서리 관측 없음")

        self.session6.observer = missing
        count = self._move_count()
        await self.session6.start()
        await asyncio.sleep(.1)
        self.assertTrue(self.session6.active)
        self.assertIsNone(self.session6.status()["arrival"])
        self.assertEqual(self._move_count(), count)
        self.assertEqual((await self.session6.stop())["state"], "STOPPED")

    async def test_stop_during_plan_blocks_send_and_other_stages(self):
        """삽입 계산 중에는 다른 명령을 차단하고 정지 이후 계산된 목표를 보내지 않는다."""
        entered, release = Event(), Event()
        original = self.session6.planner.plan

        def delayed(*args, **kwargs):
            """계획 도중 정지 요청을 넣을 시점을 확보한다."""
            entered.set()
            release.wait(3.)
            return original(*args, **kwargs)

        self.session6.planner.plan = delayed
        count = self._move_count()
        await self.session6.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 3.))
        for session in (self.console.stage1, self.console.stage2, self.console.stage3, self.console.stage4, self.console.stage5):
            with self.assertRaises(MotorError):
                await session.start()
        with self.assertRaises(MotorError):
            await self.console.move_pose({"J2": 0.})
        with self.assertRaises(MotorError):
            await self.console.scan()
        self.session6.request_stop()
        release.set()
        self.assertEqual((await self.finish())["state"], "STOPPED")
        self.assertEqual(self._move_count(), count)

    async def test_console_stop_cancels_insertion_during_observation_interval(self):
        """공통 정지 API가 6단계도 중지하고 이후 새 목표가 나오지 않게 한다."""
        sent = asyncio.Event()
        original = self.console.move_pose
        self.session6.observation_interval_s = 1.

        async def record(*args, **kwargs):
            """첫 이동을 전송한 시점을 시험에 알린다."""
            result = await original(*args, **kwargs)
            sent.set()
            return result

        self.console.move_pose = record
        count = self._move_count()
        await self.session6.start()
        await asyncio.wait_for(sent.wait(), 3.)
        await self.console.stop(tuple(ARM_JOINT_NAMES))
        status = await self.finish()
        self.assertEqual(status["state"], "STOPPED", status)
        self.assertEqual(self._move_count() - count, 1)

    async def test_unreachable_insertion_never_sends_motor_goal(self):
        """유효한 관절 목표를 만들지 못한 경우 실물 전송 없이 실패 원인을 남긴다."""
        def unreachable(*args, **kwargs):
            """관절 범위 안에서 삽입 목표를 만들지 못하는 상황을 재현한다."""
            raise ValueError("삽입 IK 해 없음")

        self.session6.planner.plan = unreachable
        count = self._move_count()
        await self.session6.start()
        status = await self.finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertIn("IK", status["message"])
        self.assertEqual(self._move_count(), count)


if __name__ == "__main__":
    unittest.main()
