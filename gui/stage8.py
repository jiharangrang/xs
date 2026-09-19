"""사용자가 뒷발 지지를 확보한 뒤 앞발을 열어 옆 빼기 없이 직진·잠금한다.
토크 해제와 지지 전환은 자동 수행하지 않고 새 깊이로 높이·정면을 함께 보정한다.
"""

import asyncio
import time

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from gui.joint_command import GRIPPER_SPEED_DEG_S, JointCommandMixin
from gui.observed_beam import ObservedBeamSession
from gui.observed_motion import ObservedMotionSettings
from hardware.camera import CameraError
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from perception.beam_edge_observation import observe_beam_edge
from planning.front_advance import DEFAULT_ADVANCE_MM, FRONT_CLOSE_DEG, FRONT_OPEN_DEG, FrontAdvancePlanner
from planning.motion_path import save_path


class FrontAdvanceRequest(BaseModel):
    """이번 실행에서 앞발을 전진시킬 거리를 밀리미터로 받는다."""

    distance_mm: float = Field(default=DEFAULT_ADVANCE_MM, gt=0, le=100, allow_inf_nan=False)


class Stage8Session(JointCommandMixin, ObservedBeamSession):
    """뒷발 지지를 유지하며 앞발 카메라로 직진 높이를 보정한 뒤 잠근다."""

    _bounded_execution = True
    _reference_stages = ()
    _observing_message = "새 깊이에서 앞발 직진 중 높이·정면 오차를 확인합니다."
    _retry_message = "새 이동 없이 빔을 다시 관측합니다"
    _reached_message = "앞발 직진·잠금각 도착 완료 · 지지 전환은 수동으로 진행해 주세요."

    def __init__(self, console, *, planner=None, observer=observe_beam_edge, settings=None, observation_interval_s=.5):
        """공통 관측 실행부와 관절 명령을 연결하며 실제 장치에는 접근하지 않는다."""
        super().__init__(console, settings or ObservedMotionSettings(timeout_s=240., motion_timeout_s=25., max_steps=200),
                         stage=8, label="앞그리퍼 전진", observation_interval_s=observation_interval_s)
        self.planner, self.observer = planner, observer
        self._holding_body = False
        self._front_open = False
        self._reference_pending = False
        self._status.update(distance_mm=DEFAULT_ADVANCE_MM, phase="waiting", fixed_tip="tip_L", motion_mode="straight",
                            support_transfer="manual", front_open_deg=FRONT_OPEN_DEG, front_close_deg=FRONT_CLOSE_DEG,
                            progress_mm=0., remaining_mm=None, height_error_mm=None, side_clearance_mm=None,
                            lateral_error_mm=None, depth_mm=None, goal_depth_mm=None, height_tolerance_mm=None, commanded_lift_mm=0.,
                            holding_goal=False, closed=False, path=None, advance_reference=None)

    async def start(self, distance_mm=DEFAULT_ADVANCE_MM):
        """수동 지지 전환 후 버튼으로 한 번의 앞발 전진을 시작한다."""
        request = FrontAdvanceRequest(distance_mm=distance_mm)
        await super().start()
        self._holding_body = False
        self._front_open = False
        self._reference_pending = False
        self._status.update(distance_mm=request.distance_mm, phase="planning", progress_mm=0., remaining_mm=None,
                            height_error_mm=None, lateral_error_mm=None, side_clearance_mm=None,
                            commanded_lift_mm=0., holding_goal=False, closed=False, path=None, advance_reference=None,
                            depth_mm=None, goal_depth_mm=None)
        return self.status()

    def _body(self, states):
        """뒷그리퍼 잠금·토크와 몸통·앞그리퍼의 실측 상태를 확인한다."""
        by_name = {state["name"]: state for state in states}
        for name in (*ARM_JOINT_NAMES, "G_L", "G_R"):
            state = by_name.get(name)
            if (state is None or state.get("error") or state.get("position_deg") is None
                    or state.get("speed_deg_s") is None
                    or not np.isfinite(state["position_deg"]) or not np.isfinite(state["speed_deg_s"])):
                raise MotorError(f"{name}: 관절 피드백을 확인해 주세요.")
            if (name == "G_L" or self._holding_body and name in ARM_JOINT_NAMES) and state.get("torque") is not True:
                raise MotorError(f"{name}: 지지를 유지할 토크가 꺼져 있습니다.")
        rear = by_name["G_L"]
        if not 0. <= rear["position_deg"] <= 5. + self.settings.tracking_tolerance_deg:
            raise MotorError("뒷그리퍼를 직접 잠그고 지지 전환을 마친 뒤 8단계를 시작해 주세요.")
        if self._front_open:
            front = by_name["G_R"]
            if front.get("torque") is not True or abs(front["position_deg"] - FRONT_OPEN_DEG) > 1.:
                raise MotorError("앞그리퍼 개방 상태를 유지하지 못했습니다.")
        return [by_name[name] for name in ARM_JOINT_NAMES]

    def _command_guard(self):
        """높이 보정과 도착 유지 명령 직전에도 뒷발 지지와 앞발 개방을 확인한다."""
        self._send_guard()

    async def _prepare_lift(self, initial):
        r"""기존 단계의 위치를 재사용하지 않고 새 전진 실행을 준비한다.

        $$\epsilon_{h,mm}=1000\epsilon_h$$
        """
        if self.planner is None:
            self.planner = await asyncio.to_thread(FrontAdvancePlanner)
        self.planner.reference = None
        self._reset_observation_cycle()
        self._status["targets_deg"] = {state["name"]: state["position_deg"] for state in initial}
        # 완료 높이 허용오차를 화면의 밀리미터 단위로 변환: $$\epsilon_{h,mm}=1000\epsilon_h$$
        self._status["height_tolerance_mm"] = 1000 * self.planner.settings.arrival_height_tolerance_m

    async def _save_preview(self, observation, states, filename, *, reset_reference=False):
        """개방 전 파지 기준을 보존하며 최신 자세의 전체 경로를 확인해 저장한다."""
        q, grips = self._angles(states), self._grippers(states)
        self._status["observation"] = observation.as_dict()
        self._logger.write("planning_input", stage=8, run_id=self._status["run_id"], filename=filename,
                           q_rad=q.tolist(), grippers_deg=grips, observation=observation.as_dict(),
                           distance_mm=self._status["distance_mm"])

        def prepare():
            """같은 계획기에서 기준 설정과 모델 경로 검사를 순서대로 수행한다."""
            if reset_reference:
                self.planner.configure(q, grips, observation, self._status["distance_mm"])
            return self.planner.preview(q, grips, observation)

        path = await asyncio.to_thread(prepare)
        saved = self._logger.path.parent / self._status["run_id"] / filename
        await asyncio.to_thread(save_path, path, saved)
        self._status.update(path=str(saved), advance_reference=self.planner.reference.as_dict())

    async def _prepare_observation(self, reader):
        """개방 전에 경로를 확인하고 현재각 유지 후 앞그리퍼만 연다."""
        while True:
            before = await self._snapshot()
            barrier = time.monotonic()
            hint = self.planner.outward_hint(self._angles(before), self._grippers(before))
            try:
                observed = await asyncio.to_thread(self.observer, reader, barrier, outward_hint=hint,
                                                   cancelled=lambda: self._cancelled)
            except (BeamDetectionError, CameraError) as error:
                await self._retry_observation(error)
                continue
            current = await self._snapshot()
            if (observed.first_frame_s <= barrier or time.monotonic() - observed.last_frame_s > .75
                    or np.max(np.abs(self._angles(current) - self._angles(before))) > np.deg2rad(.3)):
                continue
            await self._save_preview(observed, current, "before_open.json", reset_reference=True)
            fresh = await self._snapshot()
            if np.max(np.abs(self._angles(fresh) - self._angles(current))) <= np.deg2rad(.3):
                current = fresh
                break
        camera = self.planner.solver.fk.depth_camera_pose(self._angles(current))
        self._reference_world = observed.reference().transformed(camera[:3, :3], camera[:3, 3])
        targets = await self._send({s["name"]: s["position_deg"] for s in self._body(current)},
                                   "수동 안착 후 현재 실측각으로 몸통 자세를 유지합니다.", phase="holding", speed_deg_s=3.)
        self._holding_body = True
        await self._wait_targets(targets)
        targets = await self._send({"G_R": FRONT_OPEN_DEG}, "뒷발 지지를 유지하며 앞그리퍼를 -120°로 엽니다.",
                                   phase="opening", speed_deg_s=GRIPPER_SPEED_DEG_S)
        await self._wait_targets(targets)
        self._front_open = True
        self._reference_pending = True
        self._status["phase"] = "advancing"

    async def _evaluate_height(self, observation, states):
        """새 관측에서 높이와 모델 직진 목표의 도착을 확인한다."""
        if self._reference_pending:
            await self._save_preview(observation, states, "after_open.json")
            self._reference_pending = False
        q = self._angles(states)
        try:
            measured = self.planner.measure(q, self._grippers(states), observation)
        except ValueError as error:
            raise BeamDetectionError(str(error)) from error
        camera = self.planner.solver.fk.depth_camera_pose(q)
        self._reference_world = observation.reference().transformed(camera[:3, :3], camera[:3, 3])
        height_ready = abs(measured.height_error_m) <= self.planner.settings.arrival_height_tolerance_m
        forward_ready = abs(measured.forward_error_m) <= self.planner.settings.tolerance_m
        lateral_ready = abs(measured.lateral_error_m) <= self.planner.settings.tolerance_m
        reached = forward_ready and height_ready and lateral_ready
        message = "출발 횡위치를 유지하며 전진·높이·정면을 함께 보정합니다."
        if reached:
            message = "직진 목표 범위에 들어왔습니다. 현재 자세에서 도착을 확인합니다."
        elif forward_ready and lateral_ready:
            message = "전진 거리는 도착했습니다. 높이 오차를 완료 범위까지 보정합니다."
        self._update("OBSERVING", message, phase="advancing",
                     observation=observation.as_dict(), progress_mm=1000 * measured.progress_m,
                     remaining_mm=1000 * measured.forward_error_m, height_error_mm=1000 * measured.height_error_m,
                     lateral_error_mm=1000 * measured.lateral_error_m, side_clearance_mm=1000 * measured.side_clearance_m,
                     depth_mm=1000 * measured.depth_m, goal_depth_mm=1000 * measured.goal_depth_m, tilt_deg=measured.tilt_deg)
        if reached:
            return {"observed_at_s": observation.last_frame_s, "confirmation": "camera_height_and_joint_path",
                    "progress_mm": 1000 * measured.progress_m, "height_error_mm": 1000 * measured.height_error_m,
                    "lateral_error_mm": 1000 * measured.lateral_error_m, "tilt_deg": measured.tilt_deg}
        self._holding_goal = False
        self._status["holding_goal"] = False
        return None

    async def _plan_lift(self, observation, states):
        """현재 구간의 전진과 높이·정면·횡방향 보정을 함께 계산한다."""
        return await asyncio.to_thread(self.planner.plan, self._angles(states), self._grippers(states), observation)

    def _motion_options(self, step, states):
        """직진 끝에서는 속도를 줄이고 이동 중에는 다음 관측에서 목표를 갱신한다."""
        return {"speed_deg_s": 1. if step.fine else 3., "acceleration_deg_s2": 10. if step.fine else 20.}

    def _record_lift(self, step):
        """중간 관절 도착 대기 없이 다음 관측을 예약한다."""
        self._next_observation_s = time.monotonic() + self.observation_interval_s
        return "옆 빼기 없이 높이·정면을 보정하며 앞발을 직진시킵니다.", {"holding_goal": False}

    async def _complete_lift(self, arrival, states):
        """직진 목표를 연속 확인한 뒤 앞그리퍼만 잠그고 잠금각 도착을 확인한다."""
        await self._snapshot()
        self._front_open = False
        targets = await self._send({"G_R": FRONT_CLOSE_DEG}, "직진 목표 도착을 확인해 앞그리퍼를 +4.6°로 잠급니다.",
                                   phase="closing", speed_deg_s=GRIPPER_SPEED_DEG_S)
        current = await self._wait_targets(targets)
        arrival = {**arrival, "closed": True, "grip_confirmation": "joint_angle_only",
                   "positions_deg": {s["name"]: s["position_deg"] for s in current}}
        self._update("REACHED", self._reached_message, phase="closed_done", closed=True, arrival=arrival)


def install_routes(app, console):
    """수동 지지 전환 뒤 실행할 앞발 전진의 시작·중지·상태 API를 등록한다."""
    @app.get("/api/stage8")
    async def status():
        """전진과 카메라 보정·잠금 상태를 반환한다."""
        return console().stage8.status()

    @app.post("/api/stage8/start")
    async def start(request: FrontAdvanceRequest):
        """앞발 전진을 한 번 실행하며 다음 보행 단계는 자동 시작하지 않는다."""
        try:
            return await console().stage8.start(request.distance_mm)
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage8/stop")
    async def stop():
        """앞발 전진을 현재 위치에서 중지하며 추가 잠금이나 토크 해제는 하지 않는다."""
        return await console().stage8.stop()
