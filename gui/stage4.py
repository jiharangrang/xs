"""빔 아래 높이를 유지하며 고정턱이 바깥으로 빠지는 관측 기반 4단계를 실행한다.
모서리 관측·기하 계산·모터 실행을 분리하고 실제 도착 후 옆 간격을 재확인한다.
"""

import asyncio
from pathlib import Path
import time

from fastapi import HTTPException
import numpy as np

from gui.observed_motion import MotionStopped, ObservedMotionSession, ObservedMotionSettings
from hardware.camera import CameraError
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from perception.beam_edge_observation import observe_beam_edge
from planning.beam_alignment import tilt_degrees
from planning.beam_exit import BeamExitPlanner, ExitSettings
from planning.insertion_reference import (DEFAULT_RETURN_REFERENCE_PATH, load_lateral_reference,
                                          reference_from_stage4, save_lateral_reference)


class Stage4Session(ObservedMotionSession):
    """같은 빔 모서리를 추적하며 고정턱의 목표 옆 간격을 두 번 확인한다."""

    def __init__(self, console, *, planner=None, observer=observe_beam_edge, settings=None, exit_settings=None,
                 reference_path=None):
        """장치를 움직이지 않고 횡이동 설정과 표시 상태를 준비한다."""
        super().__init__(console, settings if settings is not None else ObservedMotionSettings(max_steps=30),
                         stage=4, label="외측 이동")
        self.exit_settings = exit_settings if exit_settings is not None else ExitSettings()
        self.planner = planner
        self.observer = observer
        self.reference_path = Path(reference_path) if reference_path is not None else DEFAULT_RETURN_REFERENCE_PATH
        self._status.update(clearance_mm=None, goal_clearance_mm=1000 * self.exit_settings.clearance_m,
                            vertical_gap_mm=None, commanded_exit_mm=0., displacement_world_m=None)

    def insertion_reference(self, fk):
        """현재 완료 기록을 우선하고 재시작 직후에만 저장한 복귀 기준을 읽는다."""
        if self._status["state"] == "IDLE":
            return load_lateral_reference(self.reference_path)
        return reference_from_stage4(self.status(), fk)

    async def _run(self):
        r"""정지·관측·작은 횡이동을 반복하고 관측 누락 중에는 새 명령을 보류한다.

        $$\Delta p_W=p_{tip}-p_{tip,0}$$
        """
        reader = None
        try:
            self.reference_path.unlink(missing_ok=True)
            initial = await self.console.snapshot()
            self._body(initial)
            initial_grippers = self._grippers(initial)
            self._expected_ids = {s["name"]: s.get("command_id") for s in initial}
            await self.console.run(self._guard)
            self._status.update(clearance_mm=None, vertical_gap_mm=None, commanded_exit_mm=0.,
                                displacement_world_m=None)
            if self.planner is None:
                self.planner = await asyncio.to_thread(BeamExitPlanner, settings=self.exit_settings)
            reader = self.console.camera.stream.reader()
            await asyncio.to_thread(reader.__enter__)
            deadline = time.monotonic() + self.settings.timeout_s
            self._deadline = deadline
            reference_world, start_tip = None, None
            good, nonprogress, last_clearance = 0, 0, None
            while time.monotonic() < deadline:
                states = await self._settle(deadline)
                q_before = self._angles(states)
                grippers = self._grippers(states)
                if any(abs(grippers[name] - initial_grippers[name]) > 1. for name in grippers):
                    raise MotionStopped("그리퍼 개방 상태가 달라져 외측 이동을 중지했습니다.")
                camera = self.planner.solver.fk.depth_camera_pose(q_before)
                # 월드 기준 모서리를 현재 카메라로 변환할 원점 이동: $$t_{CW}=-R_{WC}^Tp_{WC}$$
                translation = -camera[:3, :3].T @ camera[:3, 3]
                reference = None if reference_world is None else reference_world.transformed(camera[:3, :3].T, translation)
                hint = self.planner.outward_hint(q_before, grippers)
                barrier = time.monotonic()
                self._update("OBSERVING", "빔 모서리와 고정턱의 옆 간격을 확인합니다.")
                try:
                    observation = await asyncio.to_thread(self.observer, reader, barrier, outward_hint=hint,
                                                          reference=reference, cancelled=lambda: self._cancelled)
                except (BeamDetectionError, CameraError) as error:
                    good = 0
                    await self.console.run(self._guard)
                    self._update("OBSERVING", f"현재 위치에서 모서리 재관측 중 · {error}")
                    await asyncio.sleep(.2)
                    continue
                after = await self._snapshot()
                body = self._body(after)
                q_current = self._angles(after)
                if (any(s["speed_deg_s"] != 0 for s in body)
                        or np.max(np.abs(q_current - q_before)) > np.deg2rad(.3)
                        or any(abs(self._grippers(after)[name] - grippers[name]) > .3 for name in grippers)
                        or observation.first_frame_s <= barrier
                        or time.monotonic() - observation.last_frame_s > .75):
                    good = 0
                    continue
                measured = self.planner.measure(q_current, grippers, observation)
                clearance = measured.clearance_m
                tilt = tilt_degrees(observation.normal)
                self._update("OBSERVING", "고정턱 안쪽 면 기준으로 옆 간격을 확인했습니다.",
                             observation=observation.as_dict(), clearance_mm=1000 * clearance,
                             vertical_gap_mm=1000 * measured.vertical_gap_m, tilt_deg=tilt)
                if tilt > self.exit_settings.max_tilt_deg or measured.vertical_gap_m < self.exit_settings.min_vertical_gap_m:
                    raise MotionStopped("기울기 또는 빔 아래 간격이 달라졌습니다. 현재 자세를 확인해 주세요.")
                tip = self.planner.solver.fk.forward(q_current).T_world_tip_R.copy()
                if reference_world is None:
                    if measured.vertical_gap_m > self.exit_settings.max_entry_gap_m:
                        raise MotorError("먼저 3단계 첫 상승으로 빔 아래 간격을 맞춰 주세요.")
                    if self.exit_settings.clearance_m - clearance > self.exit_settings.max_total_m:
                        raise MotorError("고정턱이 빔에서 너무 멉니다. 시작 자세와 모서리를 확인해 주세요.")
                    camera = self.planner.solver.fk.depth_camera_pose(q_current)
                    reference_world = observation.reference().transformed(camera[:3, :3], camera[:3, 3])
                    start_tip = tip
                # 실측 관절 FK로 구한 이번 실행의 팁 변위: $$\Delta p_W=p_{tip}-p_{tip,0}$$
                displacement = tip[:3, 3] - start_tip[:3, 3]
                self._status["displacement_world_m"] = displacement.tolist()
                if clearance >= self.exit_settings.clearance_m - self.exit_settings.tolerance_m:
                    good += 1
                    if good >= 2:
                        arrival = {"observed_at_s": observation.last_frame_s, "clearance_mm": 1000 * clearance,
                                   "displacement_world_m": displacement.tolist(),
                                   "start_tip_world_m": start_tip[:3, 3].tolist(),
                                   "end_tip_world_m": tip[:3, 3].tolist(),
                                   "positions_deg": {s["name"]: s["position_deg"] for s in body}}
                        completed = {**self.status(), "state": "REACHED", "arrival": arrival}
                        reference = reference_from_stage4(completed, self.planner.solver.fk)
                        save_lateral_reference(reference, self.reference_path)
                        self._update("REACHED", "외측 이동 완료 · 고정턱의 옆 여유를 새 영상에서 연속 확인했습니다.",
                                     arrival=arrival)
                        return
                    continue
                good = 0
                nonprogress = nonprogress + 1 if last_clearance is not None and clearance < last_clearance + .0005 else 0
                if nonprogress >= 3:
                    raise MotorError("횡이동 후 옆 간격이 늘지 않습니다. 실제 이동과 모서리를 확인해 주세요.")
                if self._status["step"] >= self.settings.max_steps:
                    raise MotorError("외측 이동의 반복 한도에 도달했습니다.")
                step = await asyncio.to_thread(self.planner.plan, q_current, grippers, observation)
                if self._status["commanded_exit_mm"] + 1000 * step.distance_m > 1000 * self.exit_settings.max_total_m:
                    raise MotorError("이번 실행의 누적 횡이동 한도에 도달했습니다.")
                fresh = await self._snapshot()
                if (any(s["speed_deg_s"] != 0 for s in self._body(fresh))
                        or np.max(np.abs(self._angles(fresh) - q_current)) > np.deg2rad(.3)
                        or time.monotonic() - observation.last_frame_s > 1.5):
                    continue
                # 공통 모터 제어기에 전달할 도 단위 목표: $$q_{deg}=q_{rad}180/\pi$$
                targets = dict(zip(ARM_JOINT_NAMES, np.rad2deg(step.q_rad).tolist(), strict=True))
                receipt = await self.console.move_pose(targets, remember=False, owner=self, guard=self._guard)
                self._owned_ids = dict(receipt["command_ids"])
                self._expected_ids.update(self._owned_ids)
                last_clearance = clearance
                self._update("MOVING", "높이를 유지하며 바깥쪽으로 조금 이동합니다.", targets_deg=receipt["targets_deg"],
                             step=self._status["step"] + 1,
                             commanded_exit_mm=self._status["commanded_exit_mm"] + 1000 * step.distance_m)
            raise MotorError("외측 이동의 전체 대기시간이 지났습니다.")
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


def install_routes(app, console):
    """외측 이동의 시작·정지·상태 API를 등록한다."""
    @app.get("/api/stage4")
    async def status():
        """현재 옆 간격과 횡이동 진행 상태를 반환한다."""
        return console().stage4.status()

    @app.post("/api/stage4/start")
    async def start():
        """빔 아래에서 고정턱이 빠지는 외측 이동을 시작한다."""
        try:
            return await console().stage4.start()
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage4/stop")
    async def stop():
        """진행 중인 횡이동을 중지한다."""
        return await console().stage4.stop()
