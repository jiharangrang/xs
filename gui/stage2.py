"""1단계 실측 도착 뒤 관측·작은 자세 보정·재관측을 반복하는 2단계 실행부다.
공유 카메라와 공통 관절 제어기를 연결하며 정면 정렬만 수행하고 상승·삽입은 하지 않는다.
"""

import asyncio
import time

from fastapi import HTTPException
import numpy as np

from gui.joint_command import STAGE1_4_BODY_SPEED_DEG_S
from gui.observed_motion import (MotionStopped as AlignmentStopped, ObservedMotionSession,
                                 ObservedMotionSettings as Stage2Settings)
from hardware.camera import CameraError
from hardware.joint_control import JointState
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from perception.beam_observation import observe_beam
from planning.beam_alignment import BeamAlignmentPlanner, tilt_degrees


class Stage2Session(ObservedMotionSession):
    """단일 실행이 확인한 최신 관측과 관절 상태로만 다음 작은 목표를 전송한다."""

    def __init__(self, console, *, planner=None, observer=observe_beam, settings=None):
        """실제 장치를 열거나 움직이지 않고 주입 가능한 실행 구성요소를 준비한다."""
        super().__init__(console, settings if settings is not None else Stage2Settings(),
                         stage=2, label="정면 보정")
        self.planner = planner
        self.observer = observer

    def _check_entry(self, states):
        """현재 실측이 1단계 목표에 도착했는지 확인해 서버 재시작 후에도 같은 자세를 인계한다."""
        if self.console.stage1.tracker.active:
            raise MotorError("1단계 이동이 끝난 뒤 정면 보정을 시작해 주세요.")
        targets = self.console.stage1._targets()
        controller = self.console._controller
        for state in self._body(states):
            name = state["name"]
            calibration = controller._calibration(name)
            target = calibration.raw_to_degrees(calibration.degrees_to_raw(targets[name]))
            current = JointState(name, state["position_deg"], state["speed_deg_s"])
            if not controller.has_arrived(current, target):
                raise MotorError(f"{name}: 먼저 1단계 시작 자세에 도착해야 합니다.")

    async def _run(self):
        """정지 관측과 작은 보정을 반복하고 실제 영상으로만 완료를 확정한다."""
        reader = None
        try:
            initial = await self.console.snapshot()
            self._check_entry(initial)
            self._expected_ids = {state["name"]: state.get("command_id") for state in initial}
            await self.console.run(self._guard)
            reference = self._angles(initial)
            if self.planner is None:
                self.planner = await asyncio.to_thread(BeamAlignmentPlanner)
            reader = self.console.camera.stream.reader()
            await asyncio.to_thread(reader.__enter__)
            deadline = time.monotonic() + self.settings.timeout_s
            self._deadline = deadline
            good, worsening, last_tilt = 0, 0, None
            while time.monotonic() < deadline:
                states = await self._settle(deadline)
                barrier = time.monotonic()
                self._update("OBSERVING", "멈춘 자세에서 빔 기울기를 관측합니다.")
                try:
                    observation = await asyncio.to_thread(self.observer, reader, barrier,
                                                          cancelled=lambda: self._cancelled)
                except (BeamDetectionError, CameraError) as error:
                    good = 0
                    await self.console.run(self._guard)
                    self._update("OBSERVING", f"추가 이동 없이 재관측 중 · {error}")
                    await asyncio.sleep(.2)
                    continue
                after = await self._snapshot()
                body = self._body(after)
                if (any(state["speed_deg_s"] != 0 for state in body)
                        or np.max(np.abs(self._angles(after) - self._angles(states))) > np.deg2rad(.3)):
                    good = 0
                    continue
                if observation.first_frame_s <= barrier or time.monotonic() - observation.last_frame_s > .75:
                    good = 0
                    continue
                tilt = tilt_degrees(observation.normal)
                self._update("OBSERVING", f"현재 기울기 {tilt:.2f}° · 목표 {self.settings.tolerance_deg:g}° 이내",
                             observation=observation.as_dict(), tilt_deg=tilt)
                if tilt <= self.settings.tolerance_deg:
                    good += 1
                    if good >= 2:
                        self._update("ALIGNED", "정면 보정 완료 · 새 영상에서 기울기를 연속 확인했습니다.",
                                     arrival={"observed_at_s": observation.last_frame_s, "tilt_deg": tilt,
                                              "positions_deg": {state["name"]: state["position_deg"] for state in body}})
                        return
                    continue
                good = 0
                if last_tilt is not None:
                    worsening = worsening + 1 if tilt > last_tilt + 1. else 0
                    if worsening >= 2:
                        raise MotorError("보정 후 기울기가 연속으로 증가했습니다. 카메라 장착 방향을 확인해 주세요.")
                if self._status["step"] >= self.settings.max_steps:
                    raise MotorError("정면 보정 횟수 한도에 도달했습니다.")
                q_current = self._angles(after)
                step = await asyncio.to_thread(self.planner.plan, q_current, observation.normal, reference)
                fresh = await self._snapshot()
                if (any(state["speed_deg_s"] != 0 for state in self._body(fresh))
                        or np.max(np.abs(self._angles(fresh) - q_current)) > np.deg2rad(.3)
                        or time.monotonic() - observation.last_frame_s > 1.5):
                    continue
                # 계획 라디안 목표를 기존 제어기의 도 단위로 변환: $$q_{deg}=q_{rad}180/\pi$$
                degrees = np.rad2deg(step.q_rad)
                targets = dict(zip(ARM_JOINT_NAMES, degrees.tolist(), strict=True))
                receipt = await self.console.move_pose(targets, remember=False, owner=self, guard=self._guard,
                                                       speed_deg_s=STAGE1_4_BODY_SPEED_DEG_S)
                self._owned_ids = dict(receipt["command_ids"])
                self._expected_ids.update(self._owned_ids)
                last_tilt = tilt
                self._update("MOVING", "작은 자세 보정 중 · 이동 후 다시 관측합니다.",
                             step=self._status["step"] + 1, targets_deg=receipt["targets_deg"],
                             predicted_tilt_deg=step.predicted_tilt_deg)
            raise MotorError("정면 보정 대기시간이 지났습니다. 카메라 관측과 자세를 확인해 주세요.")
        except AlignmentStopped as error:
            self._update("STOPPED", str(error))
        except Exception as error:
            self._update("FAILED", str(error))
        finally:
            if self._status["state"] != "ALIGNED":
                await self._stop_owned()
            if reader is not None:
                try:
                    await asyncio.to_thread(reader.__exit__, None, None, None)
                except CameraError as error:
                    self._update(self._status["state"], self._status["message"], camera_error=str(error))


def install_routes(app, console):
    """2단계 시작·정지·상태 API를 등록하며 자동 시작은 하지 않는다."""
    @app.get("/api/stage2")
    async def status():
        """관측 기울기와 보정 실행 상태를 반환한다."""
        return console().stage2.status()

    @app.post("/api/stage2/start")
    async def start():
        """1단계 실측 도착을 확인한 뒤 정면 보정을 시작한다."""
        try:
            return await console().stage2.start()
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage2/stop")
    async def stop():
        """실행 중인 정면 보정을 종료한다."""
        return await console().stage2.stop()
