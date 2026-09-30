"""가짜 모터와 고정 공간 평면으로 간격 기반 상승의 완료·재관측·중지를 검증한다."""

import asyncio
from dataclasses import replace
import time
import unittest

import numpy as np

from gui.stage2 import Stage2Settings
from gui.stage3 import Stage3Session
from hardware.sts3215 import MotorError
from perception.beam import BeamDetectionError
from perception.beam_observation import BeamObservation
from planning.beam_lift import BeamLiftPlanner
from planning.beam_alignment import tilt_degrees
import test_gui_stage2 as stage2_tests


class Stage3ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """모터 패킷은 기존 가짜 직렬 장치에만 전달하고 카메라 거리를 움직임에 맞춰 바꾼다."""

    _q = stage2_tests.Stage2ConsoleTests._q

    async def asyncSetUp(self):
        """공통 장치 대역과 정면으로 관측되는 고정 공간 평면을 준비한다."""
        await stage2_tests.Stage2ConsoleTests.asyncSetUp(self)
        self.start_camera = self.fk.depth_camera_pose(self._q())
        self.world_normal = -self.start_camera[:3, 2]
        self.distance = .2457
        self.calls = 0
        self.console.stage3 = Stage3Session(self.console, observer=self._observe,
                                            settings=Stage2Settings(settle_s=.001, timeout_s=15.))
        self.session3 = self.console.stage3

    def _observe(self, reader, after_s, *, cancelled):
        r"""현재 카메라에서 고정 공간 평면까지의 법선·거리를 새로 계산한다.

        $$d_C=d_0+n_W^T(p_C-p_{C,0}),\quad n_C=R_{WC}^Tn_W$$
        """
        self.calls += 1
        camera = self.fk.depth_camera_pose(self._q())
        # 카메라가 이동한 뒤의 법선 좌표: $$n_C=R_{WC}^Tn_W$$
        normal = camera[:3, :3].T @ self.world_normal
        # 같은 평면까지의 이동 후 거리: $$d_C=d_0+n_W^T(p_C-p_{C,0})$$
        distance = self.distance + self.world_normal @ (camera[:3, 3] - self.start_camera[:3, 3])
        stamp = time.monotonic()
        return BeamObservation(normal, float(distance), .07, .0003, .02, stamp, stamp, 3)

    async def _finish(self):
        """진행 중인 시험 실행의 완료를 제한된 시간 동안 기다린다."""
        await asyncio.wait_for(asyncio.shield(self.session3._task), 12.)
        return self.session3.status()

    def _move_count(self):
        """가짜 버스에 전송한 묶음 이동 패킷 수를 반환한다."""
        return sum(p[4] == 0x83 for p in self.chain.packets)

    async def test_reobserved_gap_reaches_goal_and_preserves_grippers(self):
        """고정 상승량이 아니라 새로 줄어든 간격으로 완료하며 그리퍼를 유지한다."""
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        await self.session3.start()
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertLessEqual(abs(status["arrival"]["gap_mm"] - 25.), 2.)
        self.assertGreater(status["step"], 1)
        self.assertLessEqual(status["commanded_lift_mm"], 60.)
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})

    async def test_already_close_and_excessive_tilt_starts_cannot_rise(self):
        """목표보다 가까운 자세와 보정 범위를 넘는 기울기에서는 이동하지 않아요."""
        count = self._move_count()
        self.distance = .210
        await self.session3.start()
        self.assertEqual((await self._finish())["state"], "STOPPED")
        self.assertEqual(self._move_count(), count)
        self.distance = .2457
        direction = np.array([1., 0., -1.])
        direction /= np.linalg.norm(direction)
        self.world_normal = self.start_camera[:3, :3] @ direction
        await self.session3.start()
        self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(self._move_count(), count)

    async def test_entry_tilt_is_corrected_before_any_lift(self):
        r"""단계 전환 뒤 기울기가 커져도 먼저 정면을 보정한 다음 목표 간격에 도달해요.

        $$n_W=R_{WC,0}n_C$$
        """
        direction = np.array([0., .026187832868040, -.999657324975557])
        # 시작 카메라에서 관측한 약 1.5도 기울기를 월드에 고정해요: $$n_W=R_{WC,0}n_C$$
        self.world_normal = self.start_camera[:3, :3] @ direction
        planner = BeamLiftPlanner()
        original = planner.plan
        planned = []

        def record(*args):
            """각 계획의 관측 기울기와 실제로 지시한 상승량을 기록해요."""
            step = original(*args)
            planned.append((tilt_degrees(args[2]), step.kind, step.distance_m))
            return step

        planner.plan = record
        self.session3.planner = planner
        await self.session3.start()
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertEqual(planned[0][1:], ("alignment", 0.))
        self.assertTrue(any(distance > 0 for _, _, distance in planned))
        for tilt, _, distance in planned:
            if tilt > self.session3.lift_settings.alignment_tolerance_deg:
                self.assertEqual(distance, 0.)
        self.assertLessEqual(abs(status["arrival"]["gap_mm"] - 25.), 2.)

    async def test_lost_depth_holds_and_recovers(self):
        """깊이가 잠깐 사라지면 추가 상승 없이 다음 관측을 기다린다."""
        count, missed = self._move_count(), []
        original = self._observe

        def once_missing(reader, after_s, *, cancelled):
            """첫 묶음만 검출 실패로 만든다."""
            if not missed:
                missed.append(True)
                raise BeamDetectionError("깊이 누락")
            if self.calls == 0:
                self.assertEqual(self._move_count(), count)
            return original(reader, after_s, cancelled=cancelled)

        self.session3.observer = once_missing
        await self.session3.start()
        self.assertEqual((await self._finish())["state"], "REACHED")

    async def test_stage_exclusion_and_stop_while_observing(self):
        """상승 중 다른 단계를 동시에 시작하지 못하고 중지 뒤 새 명령이 나가지 않는다."""
        self.session3.settings = replace(self.session3.settings, settle_s=.3)
        await self.session3.start()
        with self.assertRaises(MotorError):
            await self.console.stage2.start()
        with self.assertRaises(MotorError):
            await self.console.stage1.start()
        count = self._move_count()
        await self.session3.stop()
        self.assertEqual(self.session3.status()["state"], "STOPPED")
        self.assertEqual(self._move_count(), count)

    async def test_no_observed_progress_prevents_repeated_blind_lifts(self):
        """명령 뒤에도 관측 간격이 줄지 않으면 무한 상승하지 않는다."""
        def unchanged(reader, after_s, *, cancelled):
            """카메라가 변하지 않은 거리만 반환하는 경우를 재현한다."""
            stamp = time.monotonic()
            return BeamObservation(np.array([0., 0., -1.]), .2457, .07, .0003, .02, stamp, stamp, 3)

        self.session3.observer = unchanged
        await self.session3.start()
        status = await self._finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertIn("간격이 줄지", status["message"])
        self.assertLessEqual(status["step"], 4)


if __name__ == "__main__":
    unittest.main()
