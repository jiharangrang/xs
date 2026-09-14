"""관절 이름과 도 단위 명령·상태를 모터의 raw 통신으로 연결하는 공통 통로다.
영점·회전 방향·감속비는 관절 설정 파일 한 곳에서 읽고, 영점 저장도 이 모듈에서 처리한다.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from hardware.sts3215 import STS3215Bus


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models/xs/calibration.yaml"
COUNTS_PER_REVOLUTION = 4096
ACCELERATION_UNIT = 100
DEFAULT_TOLERANCE_DEG = 1.0


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


def _load_calibration(path: Path) -> tuple[dict, dict[str, JointCalibration]]:
    """기존 관절 설정을 읽고 저장한 영점을 우선 적용하며 중복 ID를 거부한다."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = document["joint_order"]
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise ValueError("관절 순서가 비어 있거나 중복되었습니다.")
        result = {}
        for name in names:
            entry = document["joints"][name]
            zero_raw = entry.get("home_raw")
            if zero_raw is None:
                zero_raw = entry["home_single"]
            result[name] = JointCalibration(
                entry["servo_id"], entry["direction"], entry["gear_ratio"], zero_raw,
                math.degrees(_finite("최소 각도", entry["lower_rad"])),
                math.degrees(_finite("최대 각도", entry["upper_rad"])),
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
        *, tolerance_deg: float | None = None,
    ) -> None:
        """열린 통신 버스와 설정 파일을 연결하며 모터를 움직이거나 영점을 바꾸지 않는다."""
        tolerance = DEFAULT_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
        self._tolerance_deg = _finite("도착 허용 오차", tolerance)
        if self._tolerance_deg <= 0:
            raise ValueError("도착 허용 오차는 양수여야 합니다.")
        self._bus = bus
        self._path = Path(calibration_path)
        _, self._calibrations = _load_calibration(self._path)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """설정 파일에 정의된 관절 순서를 반환한다."""
        return tuple(self._calibrations)

    @property
    def tolerance_deg(self) -> float:
        """이 제어기를 사용하는 화면과 스크립트의 공통 도착 허용 오차를 반환한다."""
        return self._tolerance_deg

    def has_arrived(self, state: JointState, target_deg: float) -> bool:
        r"""읽은 관절 상태가 공통 위치 오차와 정지 조건을 만족하는지 판정한다.

        $$|\theta-\theta_t|\le\epsilon_\theta\;\land\;\dot\theta=0$$

        theta는 현재 각도, theta_t는 목표 각도, epsilon_theta는 도착 허용 오차다.
        한 번 읽은 상태에 대한 판정이며 모터의 불감대나 PID를 바꾸지 않는다.
        """
        position_deg = _finite("현재 각도", state.position_deg)
        target_deg = _finite("목표 각도", target_deg)
        speed_deg_s = _finite("현재 속도", state.speed_deg_s)
        # 목표와 현재 각도의 차이: $$e_\theta=|\theta-\theta_t|$$
        error_deg = abs(position_deg - target_deg)
        # 공통 도착 조건: $$e_\theta\le\epsilon_\theta\;\land\;\dot\theta=0$$
        return error_deg <= self._tolerance_deg and speed_deg_s == 0

    def _calibration(self, joint: str) -> JointCalibration:
        """알 수 없는 관절 이름을 통신 전에 거부한다."""
        if joint not in self._calibrations:
            raise ValueError(f"알 수 없는 관절입니다: {joint}")
        return self._calibrations[joint]

    def joint_for_id(self, servo_id: int) -> str:
        """초기 테스트의 ID 입력을 동일한 관절 통로로 연결한다."""
        for name, calibration in self._calibrations.items():
            if calibration.servo_id == servo_id:
                return name
        raise ValueError(f"관절 설정에 없는 모터 ID입니다: {servo_id}")

    def read(self, joint: str) -> JointState:
        """지정한 관절의 위치와 속도를 도 단위로 읽는다."""
        calibration = self._calibration(joint)
        position, speed = self._bus.read_position_speed(calibration.servo_id)
        return JointState(joint, calibration.raw_to_degrees(position), calibration.raw_speed_to_degrees(speed))

    def read_all(self) -> list[JointState]:
        """설정에 등록된 관절만 순서대로 읽으며 다른 ID를 검색하지 않는다."""
        return [self.read(joint) for joint in self.joint_names]

    def move_to(
        self, joint: str, angle_deg: float, *, speed_deg_s: float = 10.0,
        acceleration_deg_s2: float = 90.0,
    ) -> float:
        """도 단위 목표·속도·가속도를 전송하고 모터가 표현할 수 있는 목표 각도를 반환한다.

        전송 성공은 도착을 뜻하지 않으며 read로 실제 각도를 확인한다.
        """
        calibration = self._calibration(joint)
        target = calibration.degrees_to_raw(angle_deg)
        speed = calibration.speed_to_raw(speed_deg_s)
        acceleration = calibration.acceleration_to_raw(acceleration_deg_s2)
        self._bus.move_to(calibration.servo_id, target, speed=speed, acceleration=acceleration)
        return calibration.raw_to_degrees(target)

    def move_by(
        self, joint: str, delta_deg: float, *, speed_deg_s: float = 10.0,
        acceleration_deg_s2: float = 90.0,
    ) -> float:
        r"""현재 관절 각도에서 지정한 각도만큼 상대 이동한다.

        $$\theta_{\mathrm{target}}=\theta_{\mathrm{current}}+\Delta\theta$$
        """
        delta_deg = _finite("이동 각도", delta_deg)
        current = self.read(joint)
        # 현재 피드백 기준 상대 목표: $$\theta_{\mathrm{target}}=\theta_{\mathrm{current}}+\Delta\theta$$
        target_deg = current.position_deg + delta_deg
        return self.move_to(joint, target_deg, speed_deg_s=speed_deg_s, acceleration_deg_s2=acceleration_deg_s2)

    def set_torque(self, joint: str, enabled: bool) -> None:
        """지정한 관절의 토크만 켜거나 끈다."""
        self._bus.set_torque(self._calibration(joint).servo_id, enabled)

    def stop(self, joint: str) -> float:
        """현재 위치 유지 명령을 보내고 유지할 관절 각도를 도 단위로 반환한다."""
        calibration = self._calibration(joint)
        return calibration.raw_to_degrees(self._bus.stop(calibration.servo_id))

    def save_zero(self, joint: str) -> None:
        """정지한 현재 자세를 관절 영점으로 저장하며 모터 레지스터는 바꾸지 않는다.

        기존 home_single은 보존하고 home_raw에 새 기준을 저장한다.
        """
        calibration = self._calibration(joint)
        document, fresh = _load_calibration(self._path)
        if fresh != self._calibrations:
            raise ValueError("관절 설정 파일이 변경되었습니다. 다시 열고 영점을 저장해 주세요.")
        position, speed = self._bus.read_position_speed(calibration.servo_id)
        calibration.raw_to_degrees(position)
        if speed != 0:
            raise ValueError("모터가 움직이고 있습니다. 기준 자세에서 멈춘 뒤 영점을 저장해 주세요.")
        document["joints"][joint]["home_raw"] = position
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                yaml.safe_dump(document, temporary, allow_unicode=True, sort_keys=False, default_flow_style=None)
            temporary_path.replace(self._path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        _, self._calibrations = _load_calibration(self._path)
