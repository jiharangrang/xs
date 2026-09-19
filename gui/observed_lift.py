"""첫 상승과 최종 상승의 관측·계획·전송·완료 확인 순서를 공유한다.
단계별 거리 계산과 관측 시점은 각 단계에 두고 반복 실행과 명령 전송을 담당한다.
"""

import asyncio
import time

import numpy as np

from gui.observed_motion import MotionStopped, ObservedMotionSession
from hardware.camera import CameraError
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError


class ObservedLiftSession(ObservedMotionSession):
    """같은 상승 실행 순서에 단계별 관측·목표 판정·계획을 연결한다."""

    _bounded_execution = True

    async def _observation_states(self, deadline):
        """기본 상승 단계는 관절이 안정된 뒤 관측을 시작한다."""
        return await self._settle(deadline)

    async def _confirm_height(self, arrival, states):
        """단계별 높이 판정을 공통 연속 도착 확인에 넘긴다."""
        return True

    async def _retry_observation(self, error):
        """유효한 높이를 얻지 못하면 새 명령을 보내지 않고 다시 관측한다."""
        await self.console.run(self._guard)
        self._update("OBSERVING", f"{self._retry_message} · {error}")
        await asyncio.sleep(.2)

    async def _prepare_lift(self, initial):
        """단계별 계획기와 이번 실행에서 사용할 거리 이력을 준비한다."""
        raise NotImplementedError

    def _observation_options(self, states):
        """관측기에 전달할 단계별 추가 정보를 반환한다."""
        return {}

    def _feedback_unchanged(self, before, after):
        """계산 전후 관절이 멈춰 있고 같은 자세를 유지하는지 확인한다."""
        return (all(s["speed_deg_s"] == 0 for s in self._body(after))
                and np.max(np.abs(self._angles(after) - self._angles(before))) <= np.deg2rad(.3))

    async def _evaluate_height(self, observation, states):
        """현재 높이를 검사하고 목표에 들어온 경우 도착 정보를 반환한다."""
        raise NotImplementedError

    async def _plan_lift(self, observation, states):
        """단계별 높이·정면 보정 목표를 계산한다."""
        raise NotImplementedError

    def _motion_options(self, step, states):
        """전송 직전 필요한 상태를 기록하고 단계별 모터 속도 설정을 반환한다."""
        return {}

    def _record_lift(self, step):
        """전송이 끝난 이동의 거리 이력을 반영하고 표시 문구와 추가 상태를 반환한다."""
        raise NotImplementedError

    async def _run(self):
        r"""단계별 관측 시점에 작은 이동을 반복하고 새 높이 관측 두 번으로 완료한다.

        $$q_{deg}=q_{rad}180/\pi,\quad H_{next}=H+1000s$$

        H는 누적 상승량을 밀리미터로 표시한 값이고 s는 이번 계획의 미터 단위 상승량이다.
        """
        reader = None
        try:
            initial = await self.console.snapshot()
            self._body(initial)
            self._grippers(initial)
            self._expected_ids = {s["name"]: s.get("command_id") for s in initial}
            await self.console.run(self._guard)
            await self._prepare_lift(initial)
            reader = self.console.camera.stream.reader()
            await asyncio.to_thread(reader.__enter__)
            deadline = time.monotonic() + self.settings.timeout_s if self._bounded_execution else None
            self._deadline = deadline
            good = 0
            while deadline is None or time.monotonic() < deadline:
                states = await self._observation_states(deadline)
                options = self._observation_options(states)
                barrier = time.monotonic()
                self._update("OBSERVING", self._observing_message)
                try:
                    observation = await asyncio.to_thread(self.observer, reader, barrier,
                                                          cancelled=lambda: self._cancelled, **options)
                except (BeamDetectionError, CameraError) as error:
                    good = 0
                    await self._retry_observation(error)
                    continue
                after = await self._snapshot()
                if (not self._feedback_unchanged(states, after)
                        or observation.first_frame_s <= barrier
                        or time.monotonic() - observation.last_frame_s > .75):
                    good = 0
                    continue
                try:
                    arrival = await self._evaluate_height(observation, after)
                except BeamDetectionError as error:
                    good = 0
                    await self._retry_observation(error)
                    continue
                if arrival is not None:
                    if not await self._confirm_height(arrival, after):
                        good = 0
                        continue
                    good += 1
                    if good >= 2:
                        self._update("REACHED", self._reached_message, arrival=arrival)
                        return
                    continue
                good = 0
                if self._bounded_execution and self._status["step"] >= self.settings.max_steps:
                    raise MotorError(f"{self.label}의 반복 한도에 도달했습니다.")
                step = await self._plan_lift(observation, after)
                fresh = await self._snapshot()
                if (not self._feedback_unchanged(after, fresh)
                        or time.monotonic() - observation.last_frame_s > 1.5):
                    continue
                # 계획 관절각을 공통 모터 제어기의 도 단위로 변환: $$q_{deg}=q_{rad}180/\pi$$
                targets = dict(zip(ARM_JOINT_NAMES, np.rad2deg(step.q_rad).tolist(), strict=True))
                motion_options = self._motion_options(step, fresh)
                receipt = await self.console.move_pose(targets, remember=False, owner=self, guard=self._guard,
                                                        **motion_options)
                self._owned_ids = dict(receipt["command_ids"])
                self._expected_ids.update(self._owned_ids)
                message, values = self._record_lift(step)
                # 실제 전송한 계획만 누적 상승량에 반영: $$H_{next}=H+1000s$$
                total_mm = self._status["commanded_lift_mm"] + 1000 * step.distance_m
                self._update("MOVING", message, targets_deg=receipt["targets_deg"],
                             step=self._status["step"] + 1, commanded_lift_mm=total_mm, **values)
            raise MotorError(f"{self.label}의 전체 대기시간이 지났습니다.")
        except MotionStopped as error:
            self._update("STOPPED", str(error))
        except Exception as error:
            self._update("FAILED", str(error))
        finally:
            if self._status["state"] != "REACHED":
                await self._stop_owned()
            if reader is not None:
                try:
                    await asyncio.to_thread(reader.__exit__, None, None, None)
                except CameraError as error:
                    self._update(self._status["state"], self._status["message"], camera_error=str(error))
