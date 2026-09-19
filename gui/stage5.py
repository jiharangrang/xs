"""새 깊이의 높이·기울기 오차로 최종 상승과 정면 보정을 함께 실행한다.
관절 안정 대기 없이 주기적으로 목표를 갱신하고 카메라 관측으로 완료를 확인한다.
"""

import asyncio
import time

from fastapi import HTTPException

from gui.observed_beam import ObservedBeamSession
from gui.observed_motion import ObservedMotionSettings
from hardware.sts3215 import MotorError
from perception.beam import BeamDetectionError
from perception.beam_edge_observation import observe_beam_edge
from planning.beam_alignment import tilt_degrees
from planning.insertion_height import InsertionHeightPlanner, InsertionHeightSettings


class Stage5Session(ObservedBeamSession):
    """이동 중 정면을 보정하고 목표 높이를 새 관측 두 번으로 확인하면 완료한다."""

    def __init__(self, console, *, planner=None, observer=observe_beam_edge, settings=None,
                 height_settings=None, observation_interval_s=.5):
        """장치를 움직이지 않고 주기적인 높이·정면 보정 설정을 준비한다."""
        super().__init__(console, settings if settings is not None else ObservedMotionSettings(),
                         stage=5, label="최종 상승", observation_interval_s=observation_interval_s)
        self.height_settings = height_settings if height_settings is not None else (
            planner.settings if planner is not None else InsertionHeightSettings())
        self.planner = planner
        self.observer = observer
        self._status.update(remaining_mm=None, depth_mm=None, goal_depth_mm=1000 * self.height_settings.target_depth_m,
                            upper_clearance_mm=None, lower_clearance_mm=None,
                            side_clearance_mm=None, commanded_lift_mm=0., fine=False,
                            height_tolerance_mm=1000 * self.height_settings.tolerance_m,
                            tolerance_deg=self.height_settings.alignment_tolerance_deg)

    _observing_message = "새 깊이에서 목표 높이와 정면 오차를 확인합니다."
    _retry_message = "새 명령 없이 유효한 깊이를 다시 관측합니다"
    _reached_message = "최종 상승 완료 · 목표 높이를 새 관측에서 연속 확인했습니다."

    async def _prepare_lift(self, initial):
        """실행별 거리·정면 상태와 관측 주기를 초기화한다."""
        self._status.update(remaining_mm=None, depth_mm=None, goal_depth_mm=1000 * self.height_settings.target_depth_m,
                            upper_clearance_mm=None, lower_clearance_mm=None,
                            side_clearance_mm=None, commanded_lift_mm=0., fine=False,
                            motion_kind=None, holding_goal=False)
        if self.planner is None:
            self.planner = await asyncio.to_thread(InsertionHeightPlanner, settings=self.height_settings)
        self._reset_observation_cycle()

    async def _evaluate_height(self, observation, states):
        """높이와 정면 오차를 표시하고 두 목표가 함께 충족됐는지 확인한다."""
        q_current = self._angles(states)
        grippers = self._grippers(states)
        try:
            measured = self.planner.measure(q_current, grippers, observation)
            tilt = tilt_degrees(observation.normal)
        except ValueError as error:
            raise BeamDetectionError(str(error)) from error
        camera = self.planner.solver.fk.depth_camera_pose(q_current)
        self._reference_world = observation.reference().transformed(camera[:3, :3], camera[:3, 3])
        self._update("OBSERVING", "높이와 정면 오차를 같은 이동 목표에 반영합니다.",
                     observation=observation.as_dict(), remaining_mm=1000 * measured.remaining_m,
                     depth_mm=1000 * measured.depth_m, goal_depth_mm=1000 * measured.target_depth_m,
                     upper_clearance_mm=1000 * measured.upper_clearance_m,
                     lower_clearance_mm=None if measured.lower_clearance_m is None else 1000 * measured.lower_clearance_m,
                     side_clearance_mm=1000 * measured.side_clearance_m, tilt_deg=tilt)
        if abs(measured.remaining_m) <= self.height_settings.tolerance_m:
            return {"observed_at_s": observation.last_frame_s,
                    "remaining_mm": 1000 * measured.remaining_m, "tilt_deg": tilt,
                    "depth_mm": 1000 * measured.depth_m, "goal_depth_mm": 1000 * measured.target_depth_m,
                    "upper_clearance_mm": 1000 * measured.upper_clearance_m,
                    "lower_clearance_mm": None if measured.lower_clearance_m is None else 1000 * measured.lower_clearance_m,
                    "side_clearance_mm": 1000 * measured.side_clearance_m,
                    "positions_deg": {s["name"]: s["position_deg"] for s in self._body(states)}}
        self._holding_goal = False
        self._status["holding_goal"] = False
        return None

    async def _plan_lift(self, observation, states):
        """같은 새 관측으로 높이 이동과 정면 보정을 함께 계산한다."""
        return await asyncio.to_thread(self.planner.plan, self._angles(states), self._grippers(states), observation)

    def _motion_options(self, step, states):
        """목표 근처에서는 작은 이동에 맞는 기존 정밀 속도를 사용한다."""
        return {"speed_deg_s": 1. if step.fine else 3., "acceleration_deg_s2": 10. if step.fine else 30.}

    def _record_lift(self, step):
        """다음 관측 시점을 정하고 동시 높이·정면 보정 상태를 표시한다."""
        self._next_observation_s = time.monotonic() + self.observation_interval_s
        if step.kind == "alignment":
            message = "목표 높이에서 정면 오차를 보정합니다."
        elif step.kind == "lower":
            message = "높이를 낮추면서 정면을 함께 맞춥니다."
        else:
            message = "상승하면서 정면을 함께 맞춥니다."
        return message, {"motion_kind": step.kind, "fine": step.fine, "holding_goal": False}


def install_routes(app, console):
    """최종 상승의 시작·정지·상태 API를 등록한다."""
    @app.get("/api/stage5")
    async def status():
        """현재 삽입 높이와 위아래 여유를 반환한다."""
        return console().stage5.status()

    @app.post("/api/stage5/start")
    async def start():
        """고정턱이 빠진 상태에서 삽입 높이까지 상승을 시작한다."""
        try:
            return await console().stage5.start()
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage5/stop")
    async def stop():
        """진행 중인 최종 상승을 중지한다."""
        return await console().stage5.stop()
