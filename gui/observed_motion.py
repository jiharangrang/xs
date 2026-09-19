"""관측 기반 단계가 공유하는 실행 수명·관절 피드백·명령 소유권을 관리한다.
각 단계의 기하 계산과 반복 순서는 맡지 않으며 공통 모터 통신 경로를 재사용한다.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from hardware.motor_logging import MotorLogger
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES


@dataclass(frozen=True)
class ObservedMotionSettings:
    """관측·이동의 대기 한도와 실제 영상으로 확인할 완료 기준이다."""

    tolerance_deg: float = 1.
    timeout_s: float = 90.
    motion_timeout_s: float = 15.
    settle_s: float = .3
    tracking_tolerance_deg: float = 1.
    max_steps: int = 20

    def __post_init__(self):
        """반복 수와 시간·각도 설정의 유효 범위를 검사한다."""
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("최대 보정 횟수는 양의 정수여야 합니다.")
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("관측 단계 설정은 유한한 양수여야 합니다.")


class MotionStopped(RuntimeError):
    """사용자 정지나 다른 명령으로 관측 기반 실행이 중단됐음을 나타낸다."""


class ObservedMotionSession:
    """카메라를 사용하는 단계들의 피드백·직렬 전송·중지 동작을 공유한다."""

    def __init__(self, console, settings, *, stage, label):
        """장치를 움직이지 않고 공통 실행 상태와 해당 단계의 설정을 준비한다."""
        self.console = console
        self.settings = settings
        self.stage = stage
        self.label = label
        self._task = None
        self._lock = asyncio.Lock()
        self._cancelled = False
        self._expected_ids = {}
        self._owned_ids = {}
        self._logger = None
        self._deadline = None
        self._status = {"stage": stage, "state": "IDLE", "message": f"{label} 시작을 기다립니다.",
                        "run_id": None, "step": 0, "targets_deg": {}, "observation": None,
                        "tilt_deg": None, "tolerance_deg": settings.tolerance_deg,
                        "arrival": None, "log_path": None, "stop_error": None}

    def _check_start(self):
        """단계별 선행 동작의 종료 여부를 확인한다."""
        if self.console.stage1.tracker.active:
            raise MotorError("1단계 도착 후 시작해 주세요.")

    @property
    def active(self):
        """요청 연결과 무관하게 보정 작업이 진행 중인지 반환한다."""
        return self._task is not None and not self._task.done()

    def status(self):
        """화면에 전달할 현재 단계 상태의 복사본을 반환한다."""
        return {**self._status, "active": self.active}

    def _update(self, state, message, **values):
        """실행 상태와 관측·명령 이력을 함께 갱신한다."""
        self._status.update(state=state, message=message, **values)
        if self._logger is not None:
            self._logger.write("alignment", **self._status)

    def _guard(self):
        """모터 직렬 작업 안에서 정지 여부와 몸통·그리퍼 명령 소유권을 재확인한다."""
        if self._cancelled:
            raise MotionStopped("사용자가 단계 실행을 중지했습니다.")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise MotorError("단계 대기시간이 지났습니다.")
        for name, identifier in self._expected_ids.items():
            if self.console._controller.command_id(name) != identifier:
                raise MotionStopped(f"{name}: 다른 명령이 들어와 단계 실행을 중지했습니다.")

    def _body(self, states):
        """관절 피드백과 지지 그리퍼의 유지 상태가 유효한지 확인한다."""
        by_name = {state["name"]: state for state in states}
        for name in (*ARM_JOINT_NAMES, "G_L"):
            state = by_name.get(name)
            if state is None or state.get("error") or state.get("torque") is not True:
                raise MotorError(f"{name}: 관절 피드백과 토크 상태를 확인해 주세요.")
            if not np.isfinite(state["position_deg"]) or not np.isfinite(state["speed_deg_s"]):
                raise MotorError(f"{name}: 관절 상태가 유효하지 않습니다.")
        return [by_name[name] for name in ARM_JOINT_NAMES]

    def _grippers(self, states):
        """형상 계산에 필요한 두 그리퍼의 유효한 실측각을 반환한다."""
        by_name = {s["name"]: s for s in states}
        result = {}
        for name in ("G_L", "G_R"):
            state = by_name.get(name)
            if state is None or state.get("error") or not np.isfinite(state["position_deg"]):
                raise MotorError(f"{name}: 그리퍼 실측각을 확인해 주세요.")
            result[name] = state["position_deg"]
        return result

    def _angles(self, states):
        r"""최신 실측 몸통 관절각을 공통 순서의 라디안 배열로 변환한다.

        $$q_{rad}=q_{deg}\pi/180$$
        """
        # 모터의 도 단위 피드백을 기구학 입력으로 변환: $$q_{rad}=q_{deg}\pi/180$$
        return np.deg2rad([state["position_deg"] for state in self._body(states)])

    async def start(self):
        """중복 실행과 선행 단계 진행을 확인하고 독립 실행 작업을 등록한다."""
        async with self._lock:
            if self.active:
                raise MotorError(f"이미 {self.label} 중입니다.")
            self.console._require_idle_setup()
            self._check_start()
            self._cancelled = False
            self._expected_ids = {}
            self._owned_ids = {}
            self._deadline = None
            self._logger = MotorLogger(directory=Path(__file__).resolve().parents[1] / "outputs" / f"stage{self.stage}_runs")
            self._status.update(run_id=uuid4().hex, step=0, targets_deg={}, observation=None,
                                tilt_deg=None, arrival=None, stop_error=None, log_path=str(self._logger.path))
            self._update("OBSERVING", f"{self.label} 시작 조건을 확인합니다.")
            self._task = asyncio.create_task(self._run())
            return self.status()

    async def _snapshot(self):
        """현재 실행의 소유권을 확인한 새 관절 상태를 받는다."""
        states = await self.console.snapshot()
        await self.console.run(self._guard)
        self._body(states)
        return states

    def _motion_is_quiet(self, body):
        """단계의 관절 추종 허용오차 안에서 정지했는지 확인한다."""
        targets = self._status["targets_deg"]
        near = all(abs(state["position_deg"] - targets.get(state["name"], state["position_deg"]))
                   <= self.settings.tracking_tolerance_deg for state in body)
        return near and all(state["speed_deg_s"] == 0 for state in body)

    async def _settle(self, deadline):
        """작은 보정 후 충분히 멈춘 실측 상태를 기다리며 개별 예상시간 초과는 사용하지 않는다."""
        until = min(deadline, time.monotonic() + self.settings.motion_timeout_s)
        stable_since, baseline = None, None
        while time.monotonic() < until:
            states = await self._snapshot()
            body = self._body(states)
            positions = np.array([state["position_deg"] for state in body])
            quiet = self._motion_is_quiet(body)
            if not quiet or (baseline is not None and np.max(np.abs(positions - baseline)) > .3):
                stable_since, baseline = None, None
            elif stable_since is None:
                stable_since, baseline = time.monotonic(), positions
            elif time.monotonic() - stable_since >= self.settings.settle_s:
                return states
            await asyncio.sleep(.1)
        raise MotorError("보정 후 관절이 안정되지 않았습니다. 현재 위치를 확인해 주세요.")

    async def _stop_owned(self):
        """이번 보정이 전송한 뒤 다른 명령으로 바뀌지 않은 몸통 목표만 정지한다."""
        def stop():
            """소유권 확인과 정지를 하나의 직렬 작업에서 실행한다."""
            controller = self.console._controller
            names = tuple(name for name, identifier in self._owned_ids.items()
                          if controller.command_id(name) == identifier)
            self.console._apply_all(names, controller.stop)

        try:
            await self.console.run(stop)
        except (ValueError, MotorError, OSError) as error:
            self._status["stop_error"] = str(error)

    def request_stop(self):
        """외부 정지 명령 이후 보정 목표가 다시 전송되지 않도록 즉시 표시한다."""
        self._cancelled = True

    async def stop(self):
        """관측·계획·이동 어느 시점이든 중지를 요청하고 소유한 목표의 종료를 기다린다."""
        self.request_stop()
        if self.active:
            await asyncio.shield(self._task)
        return self.status()
