"""현재 빔 모서리를 보며 높이·정면을 유지하는 횡방향 삽입을 실행한다.
몸통 관절만 제어하고 4단계 출발 횡위치 복귀와 실측 높이로 완료를 판정한다.
"""

import asyncio
import time

from fastapi import HTTPException

from gui.observed_beam import ObservedBeamSession
from gui.observed_motion import ObservedMotionSettings
from hardware.sts3215 import MotorError
from perception.beam import BeamDetectionError
from perception.beam_edge_observation import observe_beam_edge
from planning.beam_insertion import BeamInsertionPlanner, InsertionSettings
from planning.insertion_height import InsertionHeightSettings


class Stage6Session(ObservedBeamSession):
    """4단계 출발 횡위치로 복귀하며 현재 빔의 높이·정면을 보정한다."""

    _reference_stages = (5,)
    _observing_message = "빔 모서리와 4단계 출발 횡위치까지의 남은 거리를 확인합니다."
    _retry_message = "새 명령 없이 같은 빔 모서리를 다시 관측합니다"
    _reached_message = "삽입 위치 확인 완료 · 4단계 출발 횡위치와 목표 높이에 도착했습니다."

    def __init__(self, console, *, planner=None, observer=observe_beam_edge, settings=None,
                 insertion_settings=None, height_settings=None, observation_interval_s=.5):
        """실물 명령 없이 삽입 계획기 설정과 표시 상태를 준비한다."""
        super().__init__(console, settings if settings is not None else ObservedMotionSettings(),
                         stage=6, label="횡방향 삽입", observation_interval_s=observation_interval_s)
        self.insertion_settings = insertion_settings if insertion_settings is not None else (
            planner.settings if planner is not None else InsertionSettings())
        self.height_settings = height_settings if height_settings is not None else (
            planner.height.settings if planner is not None else InsertionHeightSettings())
        self.planner, self.observer = planner, observer
        self._status.update(remaining_mm=None, depth_mm=None, height_error_mm=None, overlap_mm=None,
                            goal_depth_mm=1000 * self.height_settings.target_depth_m,
                            lateral_tolerance_mm=1000 * self.insertion_settings.tolerance_m,
                            height_tolerance_mm=1000 * self.insertion_settings.arrival_height_tolerance_m,
                            tolerance_deg=self.height_settings.alignment_tolerance_deg,
                            commanded_lift_mm=0., commanded_insert_mm=0., phase="aligning",
                            holding_goal=False, fine=False, motion_kind=None, lateral_reference=None)

    async def _prepare_lift(self, initial):
        """4단계 출발 기준을 고정하고 현재 자세에서 남은 거리만 복귀하도록 준비한다."""
        self._status.update(remaining_mm=None, depth_mm=None, height_error_mm=None, overlap_mm=None,
                            commanded_lift_mm=0., commanded_insert_mm=0., phase="aligning",
                            holding_goal=False, fine=False, motion_kind=None, lateral_reference=None)
        if self.planner is None:
            self.planner = await asyncio.to_thread(BeamInsertionPlanner, settings=self.insertion_settings,
                                                   height_settings=self.height_settings)
        self.planner.reference = None
        self.planner.reference = self.console.stage4.insertion_reference(self.planner.solver.fk)
        self._reset_observation_cycle()
        self._ready_to_insert = False
        self._status["lateral_reference"] = self.planner.reference.as_dict()

    async def _evaluate_height(self, observation, states):
        """삽입 준비 높이와 완료 높이를 구분하고 도착 범위에서는 재관측만 진행한다."""
        q_current = self._angles(states)
        try:
            measured = self.planner.measure(q_current, self._grippers(states), observation)
        except ValueError as error:
            raise BeamDetectionError(str(error)) from error
        camera = self.planner.solver.fk.depth_camera_pose(q_current)
        self._reference_world = observation.reference().transformed(camera[:3, :3], camera[:3, 3])
        height_ready = abs(measured.height_error_m) <= self.height_settings.tolerance_m
        if height_ready:
            self._ready_to_insert = True
        height_arrived = abs(measured.height_error_m) <= self.insertion_settings.arrival_height_tolerance_m
        reached = abs(measured.remaining_m) <= self.insertion_settings.tolerance_m and height_arrived
        phase = "inserting" if self._ready_to_insert or reached else "aligning"
        message = ("높이와 정면을 유지하며 4단계 출발 횡위치로 복귀합니다." if self._ready_to_insert
                   else "삽입 전 목표 높이를 맞추며 정면을 보정합니다.")
        if reached:
            message = "삽입 위치와 높이가 완료 범위 안입니다. 현재 자세에서 도착을 확인합니다."
        self._update("OBSERVING", message, observation=observation.as_dict(), phase=phase,
                     remaining_mm=1000 * measured.remaining_m, depth_mm=1000 * measured.depth_m,
                     height_error_mm=1000 * measured.height_error_m, overlap_mm=1000 * measured.overlap_m,
                     goal_depth_mm=1000 * measured.goal_depth_m,
                     tilt_deg=measured.tilt_deg, rgb_center_camera_m=measured.rgb_center_camera_m.tolist())
        if reached:
            return {"observed_at_s": observation.last_frame_s, "remaining_mm": 1000 * measured.remaining_m,
                    "depth_mm": 1000 * measured.depth_m, "goal_depth_mm": 1000 * measured.goal_depth_m,
                    "height_error_mm": 1000 * measured.height_error_m,
                    "tilt_deg": measured.tilt_deg, "overlap_mm": 1000 * measured.overlap_m,
                    "confirmation": "stage4_lateral_and_camera_height", "gripper_commanded": False,
                    "source_stage4_run_id": self.planner.reference.source_run_id,
                    "positions_deg": {s["name"]: s["position_deg"] for s in self._body(states)}}
        self._holding_goal = False
        self._status["holding_goal"] = False
        return None

    async def _plan_lift(self, observation, states):
        """횡방향 삽입과 기존 높이·정면 보정을 하나의 관절 목표로 계산한다."""
        return await asyncio.to_thread(self.planner.plan, self._angles(states), self._grippers(states),
                                        observation, insert_enabled=self._ready_to_insert)

    def _motion_options(self, step, states):
        """삽입은 작은 횡이동에 맞춰 낮은 속도로 실행한다."""
        return {"speed_deg_s": 1. if step.fine else 2., "acceleration_deg_s2": 10. if step.fine else 20.}

    def _record_lift(self, step):
        r"""실제 전송한 횡방향 계획량을 누적하고 다음 관측을 예약한다.

        $$S_{next}=S+1000s_y$$
        """
        self._next_observation_s = time.monotonic() + self.observation_interval_s
        # 부호 있는 삽입 계획량을 밀리미터로 누적: $$S_{next}=S+1000s_y$$
        total = self._status["commanded_insert_mm"] + 1000 * step.lateral_distance_m
        kind = "insertion" if step.lateral_distance_m > 0 else "withdraw" if step.lateral_distance_m < 0 else "alignment"
        message = "높이·정면을 함께 보정하며 4단계 출발 횡위치로 이동합니다." if step.lateral_distance_m else "삽입 높이와 정면을 맞춥니다."
        return message, {"motion_kind": kind, "fine": step.fine, "commanded_insert_mm": total,
                         "holding_goal": False}


def install_routes(app, console):
    """횡방향 삽입의 시작·정지·관측 상태 API를 등록한다."""
    @app.get("/api/stage6")
    async def status():
        """출발 횡위치까지의 남은 거리와 높이·정면 상태를 반환한다."""
        return console().stage6.status()

    @app.post("/api/stage6/start")
    async def start():
        """현재 관측에서 높이를 맞추고 4단계 출발 횡위치로 삽입한다."""
        try:
            return await console().stage6.start()
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage6/stop")
    async def stop():
        """진행 중인 삽입을 현재 자세에서 중지한다."""
        return await console().stage6.stop()
