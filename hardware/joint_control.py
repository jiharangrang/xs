"""관절 이름과 도 단위 명령·상태를 모터의 raw 통신으로 연결하는 공통 통로다.
저장된 영점·회전 방향·감속비로 변환과 제어를 수행하며 캘리브레이션 설정은 변경하지 않는다.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time

import yaml

from hardware.sts3215 import MotorError, STS3215Bus
from hardware.motor_logging import MotorLogger
from kinematics.joint_limits import DEFAULT_CALIBRATION_PATH, limits_from_document


COUNTS_PER_REVOLUTION = 4096
ACCELERATION_UNIT = 100
# 추가 감속비가 일일 때 세 카운트에 해당하는 각도 허용치: $$\epsilon_\theta=3\cdot360/N$$
DEFAULT_TOLERANCE_DEG = 3 * 360.0 / COUNTS_PER_REVOLUTION
MOVE_SETTLE_TIME_S = 2.0


def _finite(name: str, value: float) -> float:
    """입력값이 유한한 실수인지 검사한다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}에는 유한한 숫자를 입력해 주세요.")
    return float(value)


@dataclass(frozen=True)
class JointCalibration:
    """관절 하나의 영점과 방향·감속비·도 단위 범위를 보관한다."""

    servo_id: int
    direction: int
    gear_ratio: float
    zero_raw: int
    lower_deg: float
    upper_deg: float
    max_speed_deg_s: float

    def __post_init__(self) -> None:
        """명령 전송 전에 잘못된 관절 설정을 거부한다."""
        for name, value, upper in (("ID", self.servo_id, 253), ("영점", self.zero_raw, 4095)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
                raise ValueError(f"{name} 설정이 올바르지 않습니다.")
        if isinstance(self.direction, bool) or self.direction not in (-1, 1):
            raise ValueError("회전 방향은 -1 또는 1이어야 합니다.")
        if _finite("감속비", self.gear_ratio) <= 0:
            raise ValueError("감속비는 양수여야 합니다.")
        if _finite("최소 각도", self.lower_deg) >= _finite("최대 각도", self.upper_deg):
            raise ValueError("최소 각도는 최대 각도보다 작아야 합니다.")
        if _finite("최대 속도", self.max_speed_deg_s) <= 0:
            raise ValueError("최대 속도는 양수여야 합니다.")

    @property
    def counts_per_degree(self) -> float:
        r"""추가 감속비를 반영한 관절 일 도당 모터 카운트를 계산한다.

        $$c=\frac{Ng}{360}$$

        N은 모터 출력축 한 회전의 카운트, g는 관절 한 회전당 모터 출력축 회전수다.
        모터 내부 감속기는 g에 다시 포함하지 않는다.
        """
        # 관절 일 도에 대응하는 카운트: $$c=Ng/360$$
        return COUNTS_PER_REVOLUTION * self.gear_ratio / 360.0

    def raw_to_degrees(self, raw: int) -> float:
        r"""단회전 피드백을 영점 기준 관절 각도로 변환한다.

        $$\theta=d\frac{r-r_0}{c}$$

        r은 피드백, r_0는 영점, d는 방향 부호, c는 도당 카운트다.
        경계를 접거나 회전수를 추정하지 않는다.
        """
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 4095:
            raise ValueError("단회전 위치 범위를 벗어난 피드백입니다.")
        # 영점으로부터의 관절 각도: $$\theta=d(r-r_0)/c$$
        return self.direction * (raw - self.zero_raw) / self.counts_per_degree

    def degrees_to_raw(self, angle_deg: float) -> int:
        r"""관절 목표 각도를 단회전 모터 목표로 변환하고 실행 범위를 검사한다.

        $$r=\operatorname{round}(r_0+dc\theta)$$

        범위 밖 목표를 나머지 연산으로 접거나 경계값으로 자르지 않는다.
        """
        angle_deg = _finite("목표 각도", angle_deg)
        if not self.lower_deg <= angle_deg <= self.upper_deg:
            raise ValueError(f"목표 각도는 {self.lower_deg:.2f}°부터 {self.upper_deg:.2f}° 사이여야 합니다.")
        # 양자화 전 모터 목표: $$r^*=r_0+dc\theta$$
        raw_target = self.zero_raw + self.direction * self.counts_per_degree * angle_deg
        if not 0 <= raw_target <= 4095:
            raise ValueError("목표가 모터의 단회전 경계를 넘습니다. 영점과 관절 가동 범위를 확인해 주세요.")
        # 가장 가까운 엔코더 카운트로 양자화: $$r=\operatorname{round}(r^*)$$
        return round(raw_target)

    def raw_speed_to_degrees(self, raw_speed: int) -> float:
        r"""모터 카운트 속도를 관절 방향을 반영한 도 매초로 변환한다.

        $$\dot\theta=d\frac{v_r}{c}$$
        """
        # 관절 방향의 각속도: $$\dot\theta=dv_r/c$$
        return self.direction * raw_speed / self.counts_per_degree

    def speed_to_raw(self, speed_deg_s: float) -> int:
        r"""양수인 관절 이동 속도를 모터의 속도 크기로 변환한다.

        $$v_r=\operatorname{round}(c v)$$

        이동 방향은 목표 위치가 결정하므로 속도 크기에 방향 부호를 곱하지 않는다.
        """
        speed_deg_s = _finite("속도", speed_deg_s)
        if not 0 < speed_deg_s <= self.max_speed_deg_s:
            raise ValueError(f"속도는 0보다 크고 {self.max_speed_deg_s:.2f}°/s 이하여야 합니다.")
        # 속도 레지스터 크기: $$v_r=\operatorname{round}(cv)$$
        raw_speed = round(self.counts_per_degree * speed_deg_s)
        if not 1 <= raw_speed <= 3400:
            raise ValueError("입력 속도를 모터가 지원하는 속도로 표현할 수 없습니다.")
        return raw_speed

    def acceleration_to_raw(self, acceleration_deg_s2: float) -> int:
        r"""관절 가속도 크기를 모터 가속도 레지스터 단위로 변환한다.

        $$a_r=\operatorname{round}\left(\frac{c a}{100}\right)$$

        STS3215 가속도 레지스터 한 단위는 초 제곱당 백 카운트다.
        """
        acceleration_deg_s2 = _finite("가속도", acceleration_deg_s2)
        if acceleration_deg_s2 <= 0:
            raise ValueError("가속도는 양수여야 합니다.")
        # 가속도 레지스터 크기: $$a_r=\operatorname{round}(ca/100)$$
        raw_acceleration = round(self.counts_per_degree * acceleration_deg_s2 / ACCELERATION_UNIT)
        if not 1 <= raw_acceleration <= 254:
            raise ValueError("입력 가속도를 모터가 지원하는 가속도로 표현할 수 없습니다.")
        return raw_acceleration


@dataclass(frozen=True)
class JointState:
    """화면과 명령 스크립트에 전달하는 도 단위 관절 상태다."""

    name: str
    position_deg: float
    speed_deg_s: float
    target_deg: float | None = None
    error_deg: float | None = None
    tolerance_deg: float | None = None
    motion_status: str = "idle"
    torque_enabled: bool | None = None
    command_id: int | None = None
    error_raw: int | None = None
    tolerance_raw: float | None = None


@dataclass
class _JointTarget:
    """이동 명령 하나의 목표와 도착 기한 및 판정 결과를 보관한다."""

    angle_deg: float
    deadline: float
    status: str = "moving"
    error_deg: float | None = None


def load_joint_calibration(path: Path) -> tuple[dict, dict[str, JointCalibration]]:
    """기존 관절 설정을 읽고 저장한 영점을 우선 적용하며 중복 ID를 거부한다."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document.get("midpoint_pending", False), bool):
            raise ValueError("중점 설정 진행 상태가 올바르지 않습니다.")
        names = document["joint_order"]
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise ValueError("관절 순서가 비어 있거나 중복되었습니다.")
        limits = limits_from_document(document, names)
        result = {}
        for name in names:
            entry = document["joints"][name]
            zero_raw = entry.get("home_raw")
            if zero_raw is None:
                zero_raw = entry["home_single"]
            result[name] = JointCalibration(
                entry["servo_id"], entry["direction"], entry["gear_ratio"], zero_raw,
                math.degrees(limits[name][0]),
                math.degrees(limits[name][1]),
                math.degrees(_finite("최대 속도", entry["max_speed_rad_s"])),
            )
        if len({item.servo_id for item in result.values()}) != len(result):
            raise ValueError("모터 ID가 중복되었습니다.")
    except (yaml.YAMLError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"관절 설정 파일 형식이 올바르지 않습니다: {path}") from error
    return document, result


class JointController:
    """관절 명령과 피드백의 모든 단위 변환을 담당하며 raw 통신을 내부에서만 사용한다."""

    def __init__(
        self, bus: STS3215Bus, calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
        *, tolerance_deg: float | None = None, logger: MotorLogger | None = None,
    ) -> None:
        """열린 통신 버스와 설정 파일을 연결하며 모터를 움직이거나 영점을 바꾸지 않는다."""
        tolerance = DEFAULT_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
        self._tolerance_deg = _finite("도착 허용 오차", tolerance)
        self._tolerance_from_counts = tolerance_deg is None
        if self._tolerance_deg <= 0:
            raise ValueError("도착 허용 오차는 양수여야 합니다.")
        self._bus = bus
        self._path = Path(calibration_path)
        self._targets: dict[str, _JointTarget] = {}
        self._logger = logger if logger is not None else MotorLogger()
        self._motor_settings: dict[str, dict | None] = {}
        self._sample_times: dict[str, float] = {}
        self._logger.context = {**bus.connection_info, "calibration_file": str(self._path),
                                "sampling": "JointController.read 호출마다 기록",
                                "register_map": "https://docs.waveshare.net/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation/"}
        self.reload_calibration()

    @property
    def log_path(self) -> Path:
        """현재 제어기의 실행 로그 파일 경로를 반환한다."""
        return self._logger.path

    def _log_settings(self, joint: str) -> None:
        """관절별 모터 설정을 최초 한 번 기록하며 진단 조회 실패는 제어를 막지 않는다."""
        if not self._logger.enabled or joint in self._motor_settings:
            return
        self._motor_settings[joint] = None
        calibration = self._calibration(joint)
        try:
            settings = self._bus.read_settings(calibration.servo_id)
        except (MotorError, OSError, ValueError) as error:
            self._logger.write("settings_error", joint=joint, error=str(error))
        else:
            self._motor_settings[joint] = settings
            self._logger.write("settings", joint=joint, calibration=asdict(calibration), **settings)

    def set_position_gains(self, joint: str, *, p: int, i: int, d: int) -> dict:
        """관절의 위치 게인을 변경하고 변경 전후 값과 실제 재조회 설정을 로그에 남긴다."""
        self._require_reference()
        calibration = self._calibration(joint)
        self._logger.write("gain_change", joint=joint, servo_id=calibration.servo_id,
                           requested={"pid_p": p, "pid_i": i, "pid_d": d})
        try:
            result = self._bus.set_position_gains(calibration.servo_id, p=p, i=i, d=d)
        except (MotorError, OSError, ValueError) as error:
            self._logger.write("gain_change_result", joint=joint, result="failed", error=str(error))
            raise
        else:
            self._logger.write("gain_change_result", joint=joint, result="applied", **result)
            return result
        finally:
            self._motor_settings.pop(joint, None)
            self._log_settings(joint)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """설정 파일에 정의된 관절 순서를 반환한다."""
        return tuple(self._calibrations)

    @property
    def tolerance_deg(self) -> float:
        """추가 감속비를 적용하기 전 기본 각도 허용치 또는 명시한 각도 허용치를 반환한다."""
        return self._tolerance_deg

    def tolerance_for(self, joint: str) -> float:
        r"""관절별 감속비를 반영해 동일한 모터 카운트 허용치를 도 단위로 반환한다.

        $$\epsilon_\theta=\epsilon_{\mathrm{base}}/g$$

        각도 허용치를 직접 지정한 경우에는 그 값을 그대로 사용한다.
        """
        calibration = self._calibration(joint)
        if not self._tolerance_from_counts:
            return self._tolerance_deg
        # 관절 감속비를 반영한 허용 각도: $$\epsilon_\theta=\epsilon_{\mathrm{base}}/g$$
        return self._tolerance_deg / calibration.gear_ratio

    def has_arrived(self, state: JointState, target_deg: float) -> bool:
        r"""읽은 관절 상태가 공통 위치 오차와 정지 조건을 만족하는지 판정한다.

        $$|\theta-\theta_t|\le\epsilon_\theta\;\land\;\dot\theta=0$$

        theta는 현재 각도, theta_t는 목표 각도, epsilon_theta는 도착 허용 오차다.
        한 번 읽은 상태에 대한 판정이며 모터의 불감대나 PID를 바꾸지 않는다.
        """
        position_deg = _finite("현재 각도", state.position_deg)
        target_deg = _finite("목표 각도", target_deg)
        speed_deg_s = _finite("현재 속도", state.speed_deg_s)
        tolerance = self.tolerance_for(state.name)
        # 목표와 현재 각도의 차이: $$e_\theta=|\theta-\theta_t|$$
        error_deg = abs(position_deg - target_deg)
        # 공통 도착 조건: $$e_\theta\le\epsilon_\theta\;\land\;\dot\theta=0$$
        return error_deg <= tolerance and speed_deg_s == 0

    def _track_target(self, joint: str, target_deg: float, travel_time_s: float = 0.0) -> None:
        """전송한 목표를 기록하고 이동 예상 시간 뒤에 정착 시간을 추가한다."""
        deadline = time.monotonic() + travel_time_s + MOVE_SETTLE_TIME_S
        self._targets[joint] = _JointTarget(target_deg, deadline)

    def _with_motion(self, state: JointState) -> JointState:
        r"""이동 중에만 목표 오차를 확인하고 완료 또는 미도달 결과를 반환한다.

        $$e_\theta=\theta_t-\theta,\quad e_r=\operatorname{round}(dc e_\theta)$$

        판정이 끝나면 외력에 의한 변위를 감시하지 않으며 추가 명령도 보내지 않는다.
        """
        state = replace(state, command_id=self.command_id(state.name))
        target = self._targets.get(state.name)
        if target is None:
            return state
        if target.status == "moving":
            # 현재 각도에서 목표까지 남은 오차: $$e_\theta=\theta_t-\theta$$
            target.error_deg = target.angle_deg - state.position_deg
            if self.has_arrived(state, target.angle_deg):
                target.status = "arrived"
            elif time.monotonic() >= target.deadline:
                target.status = "timeout"
        calibration = self._calibration(state.name)
        # 판정 시점의 오차를 모터 카운트로 환산: $$e_r=\operatorname{round}(dc e_\theta)$$
        error_raw = round(calibration.direction * calibration.counts_per_degree * target.error_deg)
        # 관절별 허용 각도를 카운트로 환산: $$\epsilon_r=c\epsilon_\theta$$
        tolerance_raw = calibration.counts_per_degree * self.tolerance_for(state.name)
        return replace(state, target_deg=target.angle_deg, error_deg=target.error_deg,
                       tolerance_deg=self.tolerance_for(state.name), motion_status=target.status,
                       error_raw=error_raw, tolerance_raw=tolerance_raw)

    def command_id(self, joint: str) -> int | None:
        """상태가 어느 명령의 결과인지 구분할 관절별 최신 명령 번호를 반환한다."""
        self._calibration(joint)
        return self._logger.command_id(joint)

    def _calibration(self, joint: str) -> JointCalibration:
        """알 수 없는 관절 이름을 통신 전에 거부한다."""
        if joint not in self._calibrations:
            raise ValueError(f"알 수 없는 관절입니다: {joint}")
        return self._calibrations[joint]

    def _require_reference(self) -> None:
        """중점 변경과 영점 저장이 끝나지 않은 상태에서 이전 좌표를 사용하지 못하게 한다."""
        if self._midpoint_pending:
            raise MotorError("중점·영점 설정이 완료되지 않았습니다. 기준 자세에서 전체 영점 설정을 다시 실행해 주세요.")

    def reload_calibration(self) -> None:
        """캘리브레이션 모듈이 저장한 기준을 다시 읽고 실패하면 이전 기준 사용을 막는다."""
        self._targets.clear()
        self._motor_settings.clear()
        self._sample_times.clear()
        self._midpoint_pending = True
        document, calibrations = load_joint_calibration(self._path)
        self._calibrations = calibrations
        self._midpoint_pending = document.get("midpoint_pending", False)
        self._logger.write("calibration", path=str(self._path), midpoint_pending=self._midpoint_pending,
                           joints={name: asdict(value) for name, value in calibrations.items()},
                           tolerance_deg={name: self.tolerance_for(name) for name in calibrations})

    def joint_for_id(self, servo_id: int) -> str:
        """초기 테스트의 ID 입력을 동일한 관절 통로로 연결한다."""
        for name, calibration in self._calibrations.items():
            if calibration.servo_id == servo_id:
                return name
        raise ValueError(f"관절 설정에 없는 모터 ID입니다: {servo_id}")

    def read(self, joint: str) -> JointState:
        r"""관절 상태를 도 단위로 읽고 같은 응답의 전기적 피드백과 실제 목표 오차를 기록한다.

        $$e_r=r_t-r,\quad e_\theta=\theta_t-\theta$$

        기록은 읽기 호출마다 수행하며 내부에서 별도 통신 스레드를 만들지 않는다.
        """
        self._require_reference()
        calibration = self._calibration(joint)
        if not self._logger.enabled:
            position, speed = self._bus.read_position_speed(calibration.servo_id)
            state = JointState(joint, calibration.raw_to_degrees(position), calibration.raw_speed_to_degrees(speed))
            return self._with_motion(state)
        self._log_settings(joint)
        started = time.monotonic()
        feedback = None
        try:
            feedback = self._bus.read_feedback(calibration.servo_id)
            if feedback["torque_raw"] == 0:
                self._targets.pop(joint, None)
            error_bits = feedback["packet_error"] | feedback["status_raw"]
            if error_bits:
                message = feedback["packet_error_text"] or "상태 레지스터 오류"
                raise MotorError(f"모터 {calibration.servo_id} 장치 오류 0x{error_bits:02x}: {message}",
                                 device_error=error_bits)
            if feedback["torque_raw"] not in (0, 1):
                raise MotorError(f"모터 {calibration.servo_id}의 알 수 없는 토크 상태입니다: {feedback['torque_raw']}")
            state = JointState(joint, calibration.raw_to_degrees(feedback["position_raw"]),
                               calibration.raw_speed_to_degrees(feedback["speed_raw"]),
                               torque_enabled=bool(feedback["torque_raw"]))
            state = self._with_motion(state)
        except (MotorError, OSError, ValueError) as error:
            self._logger.write("sample_error", joint=joint, servo_id=calibration.servo_id,
                               command_id=self._logger.command_id(joint), read_started_s=started,
                               feedback=feedback, error=str(error),
                               communication_result=getattr(error, "communication_result", None),
                               device_error=getattr(error, "device_error", None))
            raise
        received = time.monotonic()
        previous = self._sample_times.get(joint)
        self._sample_times[joint] = received
        # 모터에 실제 저장된 목표와 피드백의 차이: $$e_r=r_t-r$$
        error_raw = feedback["goal_raw"] - feedback["position_raw"]
        goal_deg = None
        if 0 <= feedback["goal_raw"] <= 4095:
            goal_deg = calibration.raw_to_degrees(feedback["goal_raw"])
        # 모터 목표의 관절 각도 오차: $$e_\theta=\theta_t-\theta$$
        error_deg = None if goal_deg is None else goal_deg - state.position_deg
        settings = self._motor_settings.get(joint)
        current_supported = None if settings is None else not bool(settings["phase_raw"] & 0x20)
        if current_supported is False:
            feedback["current_ma"] = None
        self._logger.write("sample", joint=joint, servo_id=calibration.servo_id,
                           command_id=self._logger.command_id(joint), **feedback,
                           current_supported=current_supported, position_deg=state.position_deg,
                           speed_deg_s=state.speed_deg_s, motor_goal_deg=goal_deg,
                           motor_error_raw=error_raw, motor_error_deg=error_deg,
                           tolerance_deg=self.tolerance_for(joint), motion_status=state.motion_status,
                           command_target_deg=state.target_deg, arrival_error_deg=state.error_deg,
                           read_started_s=started, read_received_s=received,
                           read_duration_s=received - started,
                           sample_interval_s=None if previous is None else received - previous)
        return state

    def read_all(self) -> list[JointState]:
        """설정에 등록된 관절만 순서대로 읽으며 다른 ID를 검색하지 않는다."""
        return [self.read(joint) for joint in self.joint_names]

    def read_torque(self, joint: str) -> bool:
        """관절 이름으로 토크 켜짐 여부를 조회한다."""
        enabled = self._bus.read_torque(self._calibration(joint).servo_id)
        if not enabled:
            self._targets.pop(joint, None)
        return enabled

    def move_to(
        self, joint: str, angle_deg: float, *, speed_deg_s: float = 10.0,
        acceleration_deg_s2: float = 90.0,
    ) -> float:
        r"""도 단위 목표를 전송하고 피드백에서 추적할 목표와 이동 예상 시간을 기록한다.

        $$t_{\mathrm{travel}}=|\theta_t-\theta_0|/v$$

        전송 성공은 도착을 뜻하지 않으며 read로 실제 각도를 확인한다.
        """
        self._require_reference()
        calibration = self._calibration(joint)
        target = calibration.degrees_to_raw(angle_deg)
        speed = calibration.speed_to_raw(speed_deg_s)
        acceleration = calibration.acceleration_to_raw(acceleration_deg_s2)
        with self._logger.command(joint, "move_to", angle_deg=angle_deg, speed_deg_s=speed_deg_s,
                                  acceleration_deg_s2=acceleration_deg_s2, target_raw=target,
                                  speed_raw=speed, acceleration_raw=acceleration):
            current = self.read(joint)
            target_deg = calibration.raw_to_degrees(target)
            self._targets.pop(joint, None)
            self._bus.move_to(calibration.servo_id, target, speed=speed, acceleration=acceleration)
            # 지정 속도로 이동하는 데 필요한 예상 시간: $$t_{\mathrm{travel}}=|\theta_t-\theta_0|/v$$
            travel_time_s = abs(target_deg - current.position_deg) / speed_deg_s
            self._track_target(joint, target_deg, travel_time_s)
            return target_deg

    def move_many(
        self, angles_deg: dict[str, float], *, speed_deg_s: float = 10.0,
        acceleration_deg_s2: float = 90.0,
    ) -> dict[str, float]:
        r"""관절 목표 전체를 검증하고 한 패킷으로 전송하며 관절별 로그와 도착 판정을 유지한다.

        $$t_i=|\theta_{t,i}-\theta_{0,i}|/v$$

        입력에 없는 관절은 유지하며, 모든 관절이 도착할 때까지 기다리지 않고 반환한다.
        """
        self._require_reference()
        if not isinstance(angles_deg, dict) or not angles_deg:
            raise ValueError("관절 이름과 도 단위 목표 각도를 한 개 이상 지정해 주세요.")
        raw_targets = {}
        targets = {}
        for joint, angle in angles_deg.items():
            calibration = self._calibration(joint)
            try:
                position = calibration.degrees_to_raw(angle)
                speed = calibration.speed_to_raw(speed_deg_s)
                acceleration = calibration.acceleration_to_raw(acceleration_deg_s2)
            except ValueError as error:
                raise ValueError(f"{joint}: {error}") from error
            raw_targets[calibration.servo_id] = (position, speed, acceleration)
            targets[joint] = calibration.raw_to_degrees(position)
        current = {joint: self.read(joint) for joint in targets}
        with ExitStack() as commands:
            for joint, angle in angles_deg.items():
                position, speed, acceleration = raw_targets[self._calibration(joint).servo_id]
                commands.enter_context(self._logger.command(
                    joint, "move_many", angle_deg=angle, target_raw=position,
                    speed_deg_s=speed_deg_s, acceleration_deg_s2=acceleration_deg_s2,
                    speed_raw=speed, acceleration_raw=acceleration,
                ))
                self._targets.pop(joint, None)
            self._bus.move_many(raw_targets)
            for joint, target in targets.items():
                # 지정 속도로 관절별 이동 예상 시간 계산: $$t_i=|\theta_{t,i}-\theta_{0,i}|/v$$
                travel_time_s = abs(target - current[joint].position_deg) / speed_deg_s
                self._track_target(joint, target, travel_time_s)
        return targets

    def move_by(
        self, joint: str, delta_deg: float, *, speed_deg_s: float = 10.0,
        acceleration_deg_s2: float = 90.0,
    ) -> float:
        r"""현재 관절 각도에서 지정한 각도만큼 상대 이동한다.

        $$\theta_{\mathrm{target}}=\theta_{\mathrm{current}}+\Delta\theta$$
        """
        delta_deg = _finite("이동 각도", delta_deg)
        self._logger.write("relative_request", joint=joint, delta_deg=delta_deg)
        current = self.read(joint)
        # 현재 피드백 기준 상대 목표: $$\theta_{\mathrm{target}}=\theta_{\mathrm{current}}+\Delta\theta$$
        target_deg = current.position_deg + delta_deg
        return self.move_to(joint, target_deg, speed_deg_s=speed_deg_s, acceleration_deg_s2=acceleration_deg_s2)

    def set_torque(self, joint: str, enabled: bool) -> None:
        """지정한 관절의 토크를 켜거나 끄고 이전 이동 판정을 종료한다."""
        if not isinstance(enabled, bool):
            raise ValueError("토크 상태에는 True 또는 False를 사용해 주세요.")
        if enabled:
            self._require_reference()
        calibration = self._calibration(joint)
        with self._logger.command(joint, "set_torque", enabled=enabled):
            self._targets.pop(joint, None)
            self._bus.set_torque(calibration.servo_id, enabled)

    def stop(self, joint: str) -> float:
        """현재 위치 유지 명령을 보내고 유지할 관절 각도를 도 단위로 반환한다."""
        calibration = self._calibration(joint)
        with self._logger.command(joint, "stop") as result:
            self._targets.pop(joint, None)
            position = self._bus.stop(calibration.servo_id)
            result["target_raw"] = position
            self._require_reference()
            return calibration.raw_to_degrees(position)
