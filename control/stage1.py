"""1단계 자세 명령의 최신 피드백을 모아 도착 신호를 한 번 생성한다.
통신과 도착 수치 계산은 기존 관절 제어기에 맡기고 단계 상태만 관리한다.
"""

from dataclasses import asdict, dataclass
import math
import time
from uuid import uuid4

from kinematics.joints import ARM_JOINT_NAMES


@dataclass(frozen=True)
class Stage1Arrival:
    """후속 단계가 실행 식별자와 도착 이후 관측 시점을 확인할 완료 신호다."""

    run_id: str
    observed_at_s: float
    command_ids: dict[str, int]
    positions_deg: dict[str, float]


class Stage1Tracker:
    """대기·이동·도착·실패·중단을 구분하며 같은 실행의 완료 신호를 중복 생성하지 않는다."""

    def __init__(self, *, timeout_s: float = 45.0, clock=time.monotonic) -> None:
        """도착 대기 한도와 호스트 단조 시계를 준비한다."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("도착 대기 시간은 유한한 양수여야 합니다.")
        self.timeout_s = timeout_s
        self._clock = clock
        self.state = "IDLE"
        self.message = "1단계 시작을 기다립니다."
        self.run_id = None
        self.started_at_s = None
        self.command_ids = {}
        self.targets_deg = {}
        self.arrival = None

    @property
    def active(self) -> bool:
        """아직 도착을 기다리는 실행인지 반환한다."""
        return self.state == "MOVING"

    def begin(self, command_ids: dict[str, int], targets_deg: dict[str, float]) -> None:
        """전송에 성공한 몸통 관절 명령 번호와 실제 양자화 목표를 등록한다."""
        if self.active:
            raise ValueError("이미 1단계 자세로 이동 중입니다.")
        if set(command_ids) != set(ARM_JOINT_NAMES) or set(targets_deg) != set(ARM_JOINT_NAMES):
            raise ValueError("1단계에는 J1~J7 전체의 명령 번호와 목표가 필요합니다.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in command_ids.values()):
            raise ValueError("관절 명령 번호가 유효하지 않습니다.")
        if any(isinstance(value, bool) or not math.isfinite(value) for value in targets_deg.values()):
            raise ValueError("관절 목표가 유효하지 않습니다.")
        self.command_ids = dict(command_ids)
        self.targets_deg = dict(targets_deg)
        self.run_id = uuid4().hex
        self.started_at_s = self._clock()
        self.arrival = None
        self.state = "MOVING"
        self.message = "1단계 관절각의 실측 도착을 기다립니다."

    def update(self, states: list[dict]) -> Stage1Arrival | None:
        """이번 명령의 최신 관절들이 모두 현재 도착 조건을 만족하면 완료 신호를 반환한다."""
        if not self.active:
            return None
        now = self._clock()
        by_name = {state["name"]: state for state in states}
        arrived = []
        for name, command_id in self.command_ids.items():
            state = by_name.get(name)
            if state is None or state.get("error"):
                self.fail(f"{name}: 피드백을 읽지 못했습니다.")
                return None
            received_id = state.get("command_id")
            if received_id is None or received_id < command_id:
                continue
            if received_id != command_id or state.get("motion_status") == "idle" or state.get("torque") is False:
                self.cancel(f"{name}: 다른 명령 또는 토크 해제로 1단계가 중단됐습니다.")
                return None
            sampled_at = state.get("sampled_at_s")
            if sampled_at is None or not math.isfinite(sampled_at) or sampled_at < self.started_at_s:
                continue
            target = state.get("target_deg")
            if target is None or not math.isclose(target, self.targets_deg[name], rel_tol=0., abs_tol=1e-9):
                self.cancel(f"{name}: 유지 목표가 1단계 목표와 달라졌습니다.")
                return None
            if state.get("motion_status") == "timeout":
                self.fail(f"{name}: 공통 관절 제어기의 도착 제한 시간이 지났습니다.")
                return None
            if state.get("arrived_now") is True and state.get("torque") is True:
                arrived.append(state)
        if now - self.started_at_s >= self.timeout_s:
            self.fail("제한 시간 안에 모든 관절의 현재 도착을 확인하지 못했습니다.")
        elif len(arrived) == len(ARM_JOINT_NAMES):
            self.state = "ARRIVED"
            self.message = "1단계 관절각 도착을 확인했습니다. 카메라 정렬은 다음 단계입니다."
            self.arrival = Stage1Arrival(
                self.run_id, now, dict(self.command_ids),
                {state["name"]: state["position_deg"] for state in arrived},
            )
            return self.arrival
        return None

    def fail(self, message: str) -> None:
        """실측 실패를 완료와 구분해 기록한다."""
        self.state = "FAILED"
        self.message = message

    def cancel(self, message: str = "1단계를 중지했습니다.") -> None:
        """진행 중인 실행만 중단 상태로 전환한다."""
        if self.active:
            self.state = "STOPPED"
            self.message = message

    def status(self) -> dict:
        """화면과 후속 FSM이 사용할 복사된 상태와 완료 신호를 반환한다."""
        return {
            "stage": 1, "state": self.state, "message": self.message,
            "run_id": self.run_id, "started_at_s": self.started_at_s,
            "command_ids": dict(self.command_ids), "targets_deg": dict(self.targets_deg),
            "arrival": None if self.arrival is None else asdict(self.arrival),
        }
