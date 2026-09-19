"""가상 시계로 연속 목표의 주기·구간 보간·지연 건너뛰기·취소를 검증한다."""

import asyncio
from types import SimpleNamespace
import unittest

import numpy as np

from gui.continuous_motion import ContinuousJointMotion
from gui.observed_motion import MotionStopped


class VirtualClock:
    """실제로 기다리지 않고 실행 시간만 진행시키는 시험용 시계다."""

    def __init__(self):
        """시간과 요청된 대기 이력을 초기화한다."""
        self.now = 0.
        self.waits = []

    def clock(self):
        """현재 가상 시각을 반환한다."""
        return self.now

    async def sleep(self, delay):
        """요청된 대기만큼 시각을 진행하고 다른 비동기 작업에 실행을 양보한다."""
        self.waits.append(delay)
        self.now += delay
        await asyncio.sleep(0)


class ContinuousMotionTests(unittest.IsolatedAsyncioTestCase):
    """하드웨어나 도착 판정을 사용하지 않는 시간 기반 실행을 검사한다."""

    def setUp(self):
        """직선이 아닌 관절 경로와 가상 실행 시계를 준비한다."""
        self.clock = VirtualClock()
        self.motion = ContinuousJointMotion(clock=self.clock.clock, sleep=self.clock.sleep)
        self.segment = SimpleNamespace(time_s=np.array([0., 1., 2.]), duration_s=2.,
                                       q_rad=np.array([[0., 0.], [.2, -.1], [.3, -.4]]))

    async def test_time_progresses_without_arrival_and_sends_final_once(self):
        """도착 피드백 없이 주기적으로 경로를 보간하고 끝점을 정확히 한 번 전송한다."""
        self.segment.time_s *= 10
        self.segment.duration_s *= 10
        sent, progress = [], []

        async def send(q):
            """각 전송의 시각과 관절 목표를 기록한다."""
            sent.append((self.clock.now, q.copy()))
            return {"sent": len(sent)}

        receipt = await self.motion.run(self.segment, send, progress.append)
        self.assertGreater(len(sent), len(self.segment.time_s))
        self.assertEqual(receipt, {"sent": len(sent)})
        np.testing.assert_allclose(np.diff([stamp for stamp, q in sent]), .1, atol=1e-9)
        np.testing.assert_array_equal(sent[-1][1], self.segment.q_rad[-1])
        self.assertEqual(sum(np.array_equal(q, self.segment.q_rad[-1]) for _, q in sent), 1)
        self.assertTrue(np.all(np.diff(progress) >= 0))
        self.assertEqual(progress[-1], 1.)
        self.assertTrue(any(0 < q[0] < .2 and -.1 < q[1] < 0 for _, q in sent))

    async def test_slow_send_skips_missed_ticks_without_burst(self):
        """전송이 한 주기보다 오래 걸려도 밀린 목표들을 연속으로 몰아 보내지 않는다."""
        stamps = []

        async def slow_send(q):
            """첫 직렬 전송만 세 주기 가까이 지연시킨다."""
            stamps.append(self.clock.now)
            if len(stamps) == 1:
                self.clock.now += .27

        await self.motion.run(self.segment, slow_send, lambda value: None)
        self.assertAlmostEqual(stamps[1], .3)
        self.assertTrue(np.all(np.diff(stamps) >= .1 - 1e-9))
        self.assertTrue(all(delay > 0 for delay in self.clock.waits))

    async def test_stop_propagates_without_later_targets(self):
        """전송 직전 취소 검사 실패를 즉시 전달하고 이후 목표를 보내지 않는다."""
        sent = []

        async def stop_send(q):
            """세 번째 목표 전에 사용자 정지를 재현한다."""
            if len(sent) == 2:
                raise MotionStopped("시험 중지")
            sent.append(q.copy())

        with self.assertRaises(MotionStopped):
            await self.motion.run(self.segment, stop_send, lambda value: None)
        self.assertEqual(len(sent), 2)
        self.assertFalse(np.array_equal(sent[-1], self.segment.q_rad[-1]))

    def test_time_ramp_is_continuous_and_never_faster_than_planned(self):
        """출발·종료를 완만하게 연결하며 원래 경로의 최고 진행 속도를 넘지 않는다."""
        elapsed = np.linspace(0., 2.5, 1001)
        path = np.array([self.motion.path_time(t, 2.) for t in elapsed])
        self.assertEqual(path[0], 0.)
        self.assertEqual(path[-1], 2.)
        rates = np.diff(path) / np.diff(elapsed)
        self.assertGreaterEqual(float(np.min(rates)), 0.)
        self.assertLessEqual(float(np.max(rates)), 1. + 1e-10)
        self.assertLess(rates[0], .01)
        self.assertLess(rates[-1], .01)
