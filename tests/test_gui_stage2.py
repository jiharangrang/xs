"""가짜 모터와 움직임에 따라 변하는 관측으로 정면 보정의 실행·중지·상호 배제를 검증한다."""

import asyncio
from dataclasses import replace
from threading import Event
import time
import unittest
from unittest.mock import patch

import numpy as np

from gui.stage2 import Stage2Session, Stage2Settings
from hardware.motor_logging import MotorLogger
from hardware.sts3215 import MotorError
from kinematics.fk import ForwardKinematics
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from perception.beam_observation import BeamObservation
from planning.beam_alignment import BeamAlignmentPlanner
import test_gui_stage1 as stage1_tests


class DummyReader:
    """실제 카메라를 열지 않는 공유 수신 소비자 대역이다."""

    def __enter__(self):
        """장치 접근 없이 문맥에 진입한다."""
        return self

    def __exit__(self, *args):
        """종료할 외부 장치가 없음을 나타낸다."""


class Stage2ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """기존 직렬 패킷 경로를 그대로 사용해 실제 장치 없이 동작 계약을 확인한다."""

    async def asyncSetUp(self):
        """1단계 시험과 같은 버스를 만들고 새 프레임의 기하학적 관측을 연결한다."""
        await stage1_tests.Stage1ConsoleTests.asyncSetUp(self)
        self.chain.devices[0][40] = 1
        log_patch = patch("gui.observed_motion.MotorLogger", side_effect=lambda **kwargs: MotorLogger(self.pose_path.parent / "alignment"))
        log_patch.start()
        self.addCleanup(log_patch.stop)
        reader_patch = patch.object(self.console.camera.stream, "reader", side_effect=DummyReader)
        reader_patch.start()
        self.addCleanup(reader_patch.stop)
        await self.console.stage1.start()
        await self.console.stage1.observe(await self.console.snapshot())
        self.fk = ForwardKinematics()
        normal = np.array([.07, -.07, -1.])
        normal /= np.linalg.norm(normal)
        self.world_normal = self.fk.depth_camera_pose(self._q())[:3, :3] @ normal
        self.calls = 0
        settings = Stage2Settings(settle_s=.001, timeout_s=15.)
        self.console.stage2 = Stage2Session(self.console, observer=self._observe, settings=settings)
        self.session2 = self.console.stage2

    def _q(self):
        r"""가짜 엔코더 값을 실제 공통 캘리브레이션으로 관절각으로 바꾼다.

        $$q_{rad}=q_{deg}\pi/180$$
        """
        values = [self.controller._calibration(name).raw_to_degrees(
            int.from_bytes(self.chain.devices[index][56:58], "little"))
                  for index, name in enumerate(ARM_JOINT_NAMES, 1)]
        # 실측 엔코더의 각도를 라디안으로 변환: $$q_{rad}=q_{deg}\pi/180$$
        return np.deg2rad(values)

    def _observe(self, reader, after_s, *, cancelled):
        r"""움직인 카메라에서 같은 공간 평면을 새 시각에 관측한다.

        $$n_C=R_{WC}^Tn_W$$
        """
        self.calls += 1
        # 현재 카메라 회전에 따른 관측 법선: $$n_C=R_{WC}^Tn_W$$
        normal = self.fk.depth_camera_pose(self._q())[:3, :3].T @ self.world_normal
        stamp = time.monotonic()
        return BeamObservation(normal, .25, .07, .0003, .02, stamp, stamp, 3)

    async def _finish(self):
        """진행 중인 시험 보정이 끝날 때까지 제한된 시간 동안 기다린다."""
        await asyncio.wait_for(asyncio.shield(self.session2._task), timeout=12.)
        return self.session2.status()

    async def test_visual_loop_converges_and_preserves_grippers(self):
        """새 관측으로 여러 작은 보정을 수행하고 실제 기울기 기준으로만 완료한다."""
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        await self.session2.start()
        status = await self._finish()
        self.assertEqual(status["state"], "ALIGNED", status)
        self.assertLessEqual(status["arrival"]["tilt_deg"], 1.)
        self.assertGreater(status["step"], 1)
        self.assertGreaterEqual(self.calls, status["step"] + 2)
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})

    async def test_lost_detection_holds_and_reobserves(self):
        """일시적인 검출 실패 중에는 추가 명령을 보내지 않고 이후 보정을 계속한다."""
        packets = []
        original = self._observe

        def flaky(reader, after_s, *, cancelled):
            """첫 관측만 실패시킨다."""
            if not packets:
                packets.append(len(self.chain.packets))
                raise BeamDetectionError("일시적인 깊이 누락")
            if len(packets) == 1:
                self.assertEqual(len([p for p in self.chain.packets[packets[0]:] if p[4] == 0x83]), 0)
                packets.append(0)
            return original(reader, after_s, cancelled=cancelled)

        self.session2.observer = flaky
        await self.session2.start()
        self.assertEqual((await self._finish())["state"], "ALIGNED")

    async def test_entry_accepts_five_degrees_but_rejects_larger_offsets(self):
        """각 관절의 양방향 5도 경계는 허용하고 경계 밖은 거부해요."""
        states = await self.console.snapshot()
        for name in ARM_JOINT_NAMES:
            for offset in (-5., 5., -5.01, 5.01):
                with self.subTest(joint=name, offset=offset):
                    sample = [dict(state) for state in states]
                    joint = next(state for state in sample if state["name"] == name)
                    joint["position_deg"] += offset
                    if abs(offset) <= 5.:
                        self.session2._check_entry(sample)
                    else:
                        with self.assertRaisesRegex(MotorError, "5도를 넘었어요"):
                            self.session2._check_entry(sample)

    async def test_entry_still_rejects_motion_within_pose_tolerance(self):
        """자세 오차가 작아도 아직 움직이는 관절은 진입을 막아요."""
        states = await self.console.snapshot()
        joint = next(state for state in states if state["name"] == "J1")
        joint["position_deg"] += 4.
        joint["speed_deg_s"] = 1.
        with self.assertRaisesRegex(MotorError, "관절이 멈춘 뒤"):
            self.session2._check_entry(states)

    async def test_visual_loop_aligns_from_nearby_start_pose(self):
        """시작 자세에서 조금 벗어나도 현재 자세의 영상으로 정면을 보정해요."""
        calibration = self.controller._calibration("J1")
        raw = int.from_bytes(self.chain.devices[1][56:58], "little")
        shifted = calibration.degrees_to_raw(calibration.raw_to_degrees(raw) + 4.)
        self.chain.devices[1][56:58] = shifted.to_bytes(2, "little")
        await self.session2.start()
        status = await self._finish()
        self.assertEqual(status["state"], "ALIGNED", status)
        self.assertLessEqual(status["arrival"]["tilt_deg"], 1.)

    async def test_stop_and_other_commands_cannot_race_with_planned_move(self):
        """계획 계산 중 정지하거나 다른 명령을 요청해도 오래된 목표가 전송되지 않는다."""
        entered, release = Event(), Event()
        planner = BeamAlignmentPlanner()
        original = planner.plan

        def delayed(*args):
            """목표 계산이 끝나기 전에 정지 요청이 들어오도록 대기한다."""
            entered.set()
            release.wait(3.)
            return original(*args)

        planner.plan = delayed
        self.session2.planner = planner
        await self.session2.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 3.))
        with self.assertRaises(MotorError):
            await self.console.move("J1", 0., 10.)
        before = len([p for p in self.chain.packets if p[4] == 0x83])
        await self.console.stop(("J1",))
        replacement = self.controller.command_id("J1")
        release.set()
        self.assertEqual((await self._finish())["state"], "STOPPED")
        self.assertEqual(self.controller.command_id("J1"), replacement)
        self.assertEqual(len([p for p in self.chain.packets if p[4] == 0x83]), before)

    async def test_restart_can_confirm_stage1_pose_without_moving(self):
        """재시작으로 이전 완료 신호가 없어도 현재 실측 도착으로 같은 시작 자세를 인계한다."""
        self.console.stage1.tracker.state = "IDLE"
        self.console.stage1.tracker.arrival = None
        self.world_normal = self.fk.depth_camera_pose(self._q())[:3, :3] @ np.array([0., 0., -1.])
        packets = len([p for p in self.chain.packets if p[4] == 0x83])
        await self.session2.start()
        self.assertEqual((await self._finish())["state"], "ALIGNED")
        self.assertEqual(len([p for p in self.chain.packets if p[4] == 0x83]), packets)

    async def test_expired_deadline_before_send_cannot_issue_planned_target(self):
        """계산을 마친 뒤 전체 기한이 지나면 준비된 목표도 전송하지 않는다."""
        planner = BeamAlignmentPlanner()
        original = planner.plan

        def expired(*args):
            """유효한 계획을 구한 직후 실행 기한을 지난 상태로 만든다."""
            step = original(*args)
            self.session2._deadline = time.monotonic() - 1.
            return step

        planner.plan = expired
        self.session2.planner = planner
        packets = len([p for p in self.chain.packets if p[4] == 0x83])
        await self.session2.start()
        self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(len([p for p in self.chain.packets if p[4] == 0x83]), packets)

    async def test_wrong_start_pose_and_stale_observation_do_not_move(self):
        """시작 자세에서 벗어나거나 과거 영상만 제공되면 실제 보정 명령을 내지 않는다."""
        self.chain.devices[2][56:58] = (2048).to_bytes(2, "little")
        before = len([p for p in self.chain.packets if p[4] == 0x83])
        await self.session2.start()
        self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(len([p for p in self.chain.packets if p[4] == 0x83]), before)
        self.chain.devices[2][56:58] = self.chain.devices[2][42:44]
        original = self._observe

        def stale(reader, after_s, *, cancelled):
            """정지 이전 시각의 영상만 반환한다."""
            return replace(original(reader, after_s, cancelled=cancelled), first_frame_s=after_s - 1.)

        self.session2.settings = replace(self.session2.settings, timeout_s=.5)
        self.session2.observer = stale
        await self.session2.start()
        self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(len([p for p in self.chain.packets if p[4] == 0x83]), before)


if __name__ == "__main__":
    unittest.main()
