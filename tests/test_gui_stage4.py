"""가짜 모터와 고정 공간 모서리로 4단계의 완료·중지·관측 복구를 검증한다."""

import asyncio
from dataclasses import replace
from threading import Event
import time
import unittest

import numpy as np

from gui.observed_motion import ObservedMotionSettings
from gui.stage4 import Stage4Session
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from perception.beam_edge_observation import EdgeObservation
from planning.beam_exit import BeamExitPlanner
from planning.insertion_reference import LateralReference, save_lateral_reference
from test_beam_exit import GRIPPERS, REFERENCE_Q, reference_observation
import test_gui_stage1 as stage1_tests
import test_gui_stage2 as stage2_tests


class Stage4ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """기존 실제 패킷 경로를 사용하되 직렬 장치와 관측만 시험 대역으로 바꾼다."""

    _q = stage2_tests.Stage2ConsoleTests._q
    _request = stage1_tests.Stage1ConsoleTests._request

    async def asyncSetUp(self):
        r"""참고 자세의 가짜 엔코더와 카메라에 고정된 공간 모서리를 준비한다.

        $$r=\operatorname{round}(r_0+dc\theta)$$
        """
        await stage2_tests.Stage2ConsoleTests.asyncSetUp(self)
        angles = {**dict(zip(ARM_JOINT_NAMES, np.rad2deg(REFERENCE_Q), strict=True)), **GRIPPERS}
        for name, angle in angles.items():
            calibration = self.controller._calibration(name)
            # 명령 범위 검사 없이 실측 상태 대역을 구성: $$r=\operatorname{round}(r_0+dc\theta)$$
            raw = round(calibration.zero_raw + calibration.direction * calibration.counts_per_degree * angle)
            self.chain.devices[calibration.servo_id][56:58] = raw.to_bytes(2, "little")
        camera = self.fk.depth_camera_pose(self._q())
        self.reference = reference_observation().reference().transformed(camera[:3, :3], camera[:3, 3])
        self.calls = 0
        self.console.stage4 = Stage4Session(self.console, observer=self._observe,
                                            reference_path=self.pose_path.parent / "stage4_return.json",
                                            settings=ObservedMotionSettings(settle_s=.001, timeout_s=15.))
        self.session4 = self.console.stage4

    def _observe(self, reader, after_s, *, cancelled, outward_hint, reference):
        r"""움직인 카메라에서 같은 공간 모서리를 새 시각에 관측한다.

        $$p_C=R_{WC}^T(p_W-p_{WC})$$
        """
        self.calls += 1
        camera = self.fk.depth_camera_pose(self._q())
        # 현재 카메라 원점에 대한 역변환 이동량: $$t_{CW}=-R_{WC}^Tp_{WC}$$
        translation = -camera[:3, :3].T @ camera[:3, 3]
        current = self.reference.transformed(camera[:3, :3].T, translation)
        stamp = time.monotonic()
        return EdgeObservation(current.normal, current.plane_offset_m, current.width_m, .0003, .01,
                               stamp, stamp, 3, current.point_m, current.axis, current.outward, .0001,
                               reference is not None)

    async def _finish(self):
        """시험 실행의 완료를 제한된 시간 동안 기다린다."""
        await asyncio.wait_for(asyncio.shield(self.session4._task), 12.)
        return self.session4.status()

    def _move_count(self):
        """가짜 버스로 전송한 묶음 이동 패킷 수를 반환한다."""
        return sum(packet[4] == 0x83 for packet in self.chain.packets)

    async def test_reobserved_clearance_and_cartesian_return_record(self):
        """새 모서리 관측으로 완료하고 실제 팁 변위를 남기며 두 그리퍼를 유지한다."""
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        code, response = await self._request("/api/stage4/start")
        self.assertEqual(code, 200)
        status = await self._finish()
        self.assertEqual(status["state"], "REACHED", status)
        self.assertAlmostEqual(status["clearance_mm"], 10., delta=1.)
        self.assertGreater(status["step"], 1)
        self.assertTrue(status["observation"]["single_edge"])
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})
        self.assertGreater(np.linalg.norm(status["arrival"]["displacement_world_m"]), .015)
        code, current = await self._request("/api/stage4", "GET")
        self.assertEqual(current["arrival"], status["arrival"])
        reference = self.session4.insertion_reference(self.fk)
        restarted = Stage4Session(self.console, reference_path=self.session4.reference_path)
        self.assertEqual(restarted.insertion_reference(self.fk), reference)
        self.assertEqual(reference.source_run_id, status["run_id"])
        self.assertAlmostEqual(reference.remaining(status["arrival"]["start_tip_world_m"]), 0.)

    async def test_detection_loss_holds_then_recovers(self):
        """관측 실패 동안 추가 명령을 보내지 않고 다음 새 관측에서 계속한다."""
        original, missed = self._observe, []
        count = self._move_count()

        def intermittent(*args, **kwargs):
            """첫 관측 묶음만 일시적으로 누락시킨다."""
            if not missed:
                missed.append(True)
                raise BeamDetectionError("일시적 모서리 누락")
            if self.calls == 0:
                self.assertEqual(self._move_count(), count)
            return original(*args, **kwargs)

        self.session4.observer = intermittent
        await self.session4.start()
        self.assertEqual((await self._finish())["state"], "REACHED")

    async def test_no_progress_stops_before_unbounded_exit(self):
        """모서리 관측이 명령에 반응하지 않으면 작은 이동 한도 안에서 중지한다."""
        initial = self._observe(None, 0., cancelled=lambda: False, outward_hint=None, reference=None)

        def unchanged(*args, **kwargs):
            """시각만 새로운 고정된 카메라 관측을 반환한다."""
            stamp = time.monotonic()
            return replace(initial, first_frame_s=stamp, last_frame_s=stamp)

        self.session4.observer = unchanged
        await self.session4.start()
        status = await self._finish()
        self.assertEqual(status["state"], "FAILED", status)
        self.assertIn("옆 간격이 늘지", status["message"])
        self.assertLessEqual(status["step"], 4)

    async def test_entry_height_is_checked_without_relying_on_old_stage_status(self):
        """오래된 단계 완료 표시 대신 새로 관측한 빔 아래 높이로 진입을 판단한다."""
        count = self._move_count()
        old = LateralReference((0., 0., 0.), (0., 1., 0.), "old-run")
        save_lateral_reference(old, self.session4.reference_path)
        self.reference = replace(self.reference, plane_offset_m=self.reference.plane_offset_m + .03)
        await self.session4.start()
        self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(self._move_count(), count)
        self.assertFalse(self.session4.reference_path.exists())
        with self.assertRaises(ValueError):
            self.session4.insertion_reference(self.fk)

    async def test_stop_during_planning_prevents_transmission_and_excludes_other_stages(self):
        """계획 중에는 다른 이동을 막고 정지 요청 뒤 계산된 목표를 폐기한다."""
        entered, release = Event(), Event()
        planner = BeamExitPlanner()
        original = planner.plan

        def delayed(*args):
            """목표 계산 중 정지 요청을 넣을 시간을 확보한다."""
            entered.set()
            release.wait(3.)
            return original(*args)

        planner.plan = delayed
        self.session4.planner = planner
        count = self._move_count()
        await self.session4.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 3.))
        for session in (self.console.stage1, self.console.stage2, self.console.stage3):
            with self.assertRaises(MotorError):
                await session.start()
        self.session4.request_stop()
        release.set()
        self.assertEqual((await self._finish())["state"], "STOPPED")
        self.assertEqual(self._move_count(), count)

    async def test_stale_observations_cannot_move_or_complete(self):
        """시각이 오래된 관측은 도착이나 다음 이동의 근거로 사용하지 않는다."""
        original = self._observe

        def stale(*args, **kwargs):
            """이동 이전 시각으로 표시된 관측을 반환한다."""
            return replace(original(*args, **kwargs), first_frame_s=0., last_frame_s=0.)

        self.session4.observer = stale
        self.session4.settings = replace(self.session4.settings, timeout_s=.4)
        count = self._move_count()
        await self.session4.start()
        self.assertEqual((await self._finish())["state"], "FAILED")
        self.assertEqual(self._move_count(), count)


if __name__ == "__main__":
    unittest.main()
