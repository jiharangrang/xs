"""새 깊이로 빔과의 간격을 확인하고 정면을 보정한 뒤 목표 간격까지 상승해요.
관측과 실제 도착 대기는 공통 실행부를 사용해요.
"""

import asyncio

from fastapi import HTTPException

from gui.joint_command import STAGE1_4_BODY_SPEED_DEG_S
from gui.observed_lift import ObservedLiftSession
from gui.observed_motion import MotionStopped, ObservedMotionSettings
from hardware.sts3215 import MotorError
from perception.beam_observation import observe_beam
from planning.beam_alignment import tilt_degrees
from planning.beam_lift import BeamLiftPlanner, LiftSettings


class Stage3Session(ObservedLiftSession):
    """매번 관측한 간격으로 다음 상승량을 정하고 새 관측 두 번으로 완료를 확인한다."""

    def __init__(self, console, *, planner=None, observer=observe_beam, settings=None, lift_settings=None):
        """공통 실행부와 간격 목표를 준비하며 실제 장치에는 명령하지 않는다."""
        super().__init__(console, settings if settings is not None else ObservedMotionSettings(max_steps=30),
                         stage=3, label="첫 상승")
        self.planner = planner
        self.observer = observer
        self.lift_settings = lift_settings if lift_settings is not None else LiftSettings()
        self._status.update(gap_mm=None, goal_gap_mm=1000 * self.lift_settings.gap_m, commanded_lift_mm=0.)

    _observing_message = "새 깊이에서 빔과 그리퍼 사이 간격을 확인합니다."
    _retry_message = "추가 상승 없이 재관측 중"
    _reached_message = "첫 상승 완료 · 목표 간격을 새 깊이에서 연속 확인했습니다."

    async def _prepare_lift(self, initial):
        """첫 상승의 목표 간격과 진행 이력을 초기화한다."""
        self._status.update(gap_mm=None, commanded_lift_mm=0.)
        if self.planner is None:
            self.planner = await asyncio.to_thread(BeamLiftPlanner, settings=self.lift_settings)
        self._initial_gap = None
        self._last_gap = None
        self._nonprogress = 0

    def _motion_options(self, step, states):
        """첫 상승과 정면 재보정에 1~4단계 공통 몸통 속도를 적용해요."""
        return {"speed_deg_s": STAGE1_4_BODY_SPEED_DEG_S}

    async def _evaluate_height(self, observation, states):
        """시작 거리와 과상승을 확인하고 기울기가 남으면 기존 보정 계획으로 넘겨요."""
        q_current = self._angles(states)
        grippers = self._grippers(states)
        tilt = tilt_degrees(observation.normal)
        gap = await asyncio.to_thread(self.planner.gap, q_current, grippers,
                                       observation.normal, observation.plane_offset_m)
        self._gap = gap
        self._update("OBSERVING", "빔 아래 간격을 확인했습니다.", observation=observation.as_dict(),
                     gap_mm=1000 * gap, tilt_deg=tilt)
        if self._initial_gap is None:
            if gap - self.lift_settings.gap_m > self.lift_settings.max_total_m:
                raise MotorError("목표 간격까지 너무 멉니다. 시작 자세를 확인해 주세요.")
            self._initial_gap = gap
        if gap < self.lift_settings.gap_m - self.lift_settings.tolerance_m:
            raise MotionStopped("목표보다 빔에 가까워 추가 상승을 중지했습니다. 실제 간격을 확인해 주세요.")
        if gap <= self.lift_settings.gap_m + self.lift_settings.tolerance_m and tilt <= self.lift_settings.alignment_tolerance_deg:
            return {"observed_at_s": observation.last_frame_s, "gap_mm": 1000 * gap,
                    "positions_deg": {s["name"]: s["position_deg"] for s in self._body(states)}}
        return None

    async def _plan_lift(self, observation, states):
        """기존 첫 상승 계획을 호출하고 간격 감소와 누적 상승 한도를 확인한다."""
        step = await asyncio.to_thread(self.planner.plan, self._angles(states), self._grippers(states),
                                        observation.normal, observation.plane_offset_m)
        if step.distance_m > 0:
            self._nonprogress = self._nonprogress + 1 if self._last_gap is not None and self._gap >= self._last_gap - .001 else 0
            if self._nonprogress >= 3:
                raise MotorError("상승 후 간격이 줄지 않습니다. 모터 이동과 카메라 방향을 확인해 주세요.")
            if self._status["commanded_lift_mm"] + step.distance_m * 1000 > self.lift_settings.max_total_m * 1000:
                raise MotorError("이번 실행의 누적 상승 한도에 도달했습니다.")
        return step

    def _record_lift(self, step):
        """전송한 상승만 다음 간격 감소 판정의 기준으로 저장한다."""
        self._last_gap = self._gap if step.distance_m > 0 else None
        message = "작은 상승 후 다시 간격을 측정합니다." if step.kind == "lift" else "현재 높이에서 정면을 다시 맞춥니다."
        return message, {}


def install_routes(app, console):
    """첫 상승의 시작·정지·상태 API를 등록한다."""
    @app.get("/api/stage3")
    async def status():
        """현재 간격과 상승 진행 상태를 반환한다."""
        return console().stage3.status()

    @app.post("/api/stage3/start")
    async def start():
        """새 정면 관측으로 확인한 자세에서 목표 간격까지 상승을 시작한다."""
        try:
            return await console().stage3.start()
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage3/stop")
    async def stop():
        """진행 중인 상승을 중지한다."""
        return await console().stage3.stop()
