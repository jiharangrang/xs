"""제조사 SDK로 STS3215 모터의 위치를 읽고 단회전 위치 명령을 보낸다.
관절 영점 변환 없이 모터 원시값을 사용하며, 한 통신 포트는 한 실행 흐름에서 사용한다.
"""

import time
from typing import Self

from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import (
    SMS_STS_MIN_ANGLE_LIMIT_L,
    SMS_STS_MODE,
    SMS_STS_TORQUE_ENABLE,
)

MIDPOINT_RAW = 2048


class MotorError(RuntimeError):
    """모터의 통신 실패 또는 장치 오류를 나타낸다."""

    def __init__(self, message: str, *, communication_result: int | None = None,
                 device_error: int | None = None) -> None:
        """오류 설명과 SDK 통신 결과를 보관해 무응답과 다른 실패를 구분한다."""
        super().__init__(message)
        self.communication_result = communication_result
        self.device_error = device_error


def _check_integer(name: str, value: int, lower: int, upper: int) -> None:
    """모터 명령에 사용할 정수가 지정 범위에 있는지 검사한다."""
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name}은 {lower}부터 {upper} 사이의 정수여야 합니다.")


class STS3215Bus:
    """직렬 통신 포트 하나를 통해 지정한 모터에 명령을 전달한다."""

    def __init__(self, port: str, baudrate: int = 1_000_000) -> None:
        """포트 이름과 통신 속도를 저장하고 실제 연결은 열기 호출까지 미룬다."""
        self._port = PortHandler(port)
        self._sdk = sms_sts(self._port)
        self._baudrate = baudrate

    def open(self) -> Self:
        """통신 포트를 열고 모터 설정이나 토크 상태는 바꾸지 않는다."""
        if self._port.is_open:
            return self
        try:
            if not self._port.setBaudRate(self._baudrate):
                raise ValueError(f"SDK가 지원하지 않는 통신 속도입니다: {self._baudrate}")
            self._port.ser.write_timeout = 0.5
        except (OSError, ValueError):
            self.close()
            raise
        return self

    def close(self) -> None:
        """통신 포트만 닫으며 모터의 목표 위치와 토크 상태는 유지한다."""
        if self._port.is_open:
            self._port.closePort()

    def __enter__(self) -> Self:
        """문맥 관리자 진입 시 통신 포트를 연다."""
        return self.open()

    def __exit__(self, *args: object) -> None:
        """문맥 관리자 종료 시 통신 포트를 닫는다."""
        self.close()

    def _call(self, servo_id: int, method: str, *args: int,
              include_device_error: bool = False) -> list:
        """SDK 호출 결과에서 통신 오류와 모터 오류를 확인하고 읽은 값을 반환한다."""
        _check_integer("모터 ID", servo_id, 0, 253)
        if not self._port.is_open:
            raise MotorError("통신 포트를 먼저 열어 주세요.")
        try:
            *values, result, error = getattr(self._sdk, method)(servo_id, *args)
        except (OSError, IndexError) as error:
            raise MotorError(f"모터 {servo_id} 통신 실패 ({method}): {error}") from error
        finally:
            self._port.is_using = False
        if result != COMM_SUCCESS:
            raise MotorError(
                f"모터 {servo_id} 응답 실패 ({method}): {self._sdk.getTxRxResult(result)}",
                communication_result=result,
            )
        if error and not include_device_error:
            raise MotorError(
                f"모터 {servo_id} 장치 오류 0x{error:02x}: {self._sdk.getRxPacketError(error)}",
                device_error=error,
            )
        if include_device_error:
            values.append(error)
        return values

    @property
    def connection_info(self) -> dict:
        """로그에 사용할 포트와 통신 속도를 반환한다."""
        return {"port": self._port.port_name, "baudrate": self._baudrate}

    def read_register_block(self, servo_id: int, address: int, size: int) -> tuple[bytes, int]:
        """진단용 연속 레지스터와 장치 오류를 함께 읽고 손상·누락 응답은 거부한다."""
        _check_integer("레지스터 주소", address, 0, 255)
        _check_integer("읽기 길이", size, 1, 128)
        if address + size > 256:
            raise ValueError("레지스터 읽기 범위를 벗어났습니다.")
        data, error = self._call(servo_id, "readTxRx", address, size, include_device_error=True)
        if len(data) != size:
            raise MotorError(f"모터 {servo_id} 진단 응답 길이가 올바르지 않습니다.", device_error=error)
        return bytes(data), error

    def read_feedback(self, servo_id: int) -> dict:
        r"""동작 상태와 전기적 피드백을 한 번에 읽으며 장치 오류도 진단값에 보존한다.

        $$p_{\mathrm{PWM}}=L/10,\quad p_{\mathrm{limit}}=L_{\max}/10,\quad
        I_{\mathrm{mA}}=6.5I_r,\quad V=V_r/10$$

        L은 방향 부호를 해석한 부하 값이며 실제 출력축 토크를 뜻하지 않는다.
        전류는 제조사 표의 전류 피드백 단위를 사용한다.
        """
        data, error = self.read_register_block(servo_id, 40, 31)
        word = lambda offset: int.from_bytes(data[offset:offset + 2], "little")
        load_raw = self._sdk.scs_tohost(word(20), 10)
        current_raw = word(29)
        # 부하 피드백을 PWM 백분율로 변환: $$p_{\mathrm{PWM}}=L/10$$
        pwm_percent = load_raw / 10.0
        # 전류 피드백을 밀리암페어로 변환: $$I_{\mathrm{mA}}=6.5I_r$$
        current_ma = current_raw * 6.5
        # 전압 피드백을 볼트로 변환: $$V=V_r/10$$
        voltage_v = data[22] / 10.0
        limit_raw = word(8)
        # 현재 설정된 출력 제한을 백분율로 변환: $$p_{\mathrm{limit}}=L_{\max}/10$$
        output_limit_percent = limit_raw / 10.0
        return {
            "position_raw": self._sdk.scs_tohost(word(16), 15),
            "speed_raw": self._sdk.scs_tohost(word(18), 15),
            "goal_raw": self._sdk.scs_tohost(word(2), 15),
            "goal_feedback_raw": self._sdk.scs_tohost(word(27), 15),
            "goal_speed_raw": self._sdk.scs_tohost(word(6), 15),
            "acceleration_raw": data[1], "torque_raw": data[0],
            "output_limit_raw": limit_raw, "output_limit_percent": output_limit_percent,
            "output_at_limit": abs(load_raw) >= limit_raw if 0 < limit_raw <= 1000 else None,
            "load_raw": load_raw,
            "pwm_percent": pwm_percent, "current_raw": current_raw,
            "current_ma": current_ma, "voltage_v": voltage_v, "temperature_c": data[23],
            "moving_raw": data[26], "status_raw": data[25], "packet_error": error,
            "packet_error_text": self._sdk.getRxPacketError(error) if error else None,
            "registers_40_70_hex": data.hex(),
        }

    def read_settings(self, servo_id: int) -> dict:
        """모델·PID·불감대·보호 설정을 읽기 전용으로 한 번에 조회한다."""
        data, error = self.read_register_block(servo_id, 0, 40)
        word = lambda offset: int.from_bytes(data[offset:offset + 2], "little")
        return {
            "firmware_major": data[0], "firmware_minor": data[1], "model_number": word(3),
            "servo_id": data[5], "baudrate_code": data[6], "response_level": data[8],
            "min_position_raw": word(9), "max_position_raw": word(11),
            "max_temperature_c": data[13], "max_voltage_raw": data[14], "min_voltage_raw": data[15],
            "max_output_raw": word(16), "phase_raw": data[18],
            "protection_mask": data[19], "alarm_mask": data[20],
            "pid_p": data[21], "pid_d": data[22], "pid_i": data[23],
            "startup_output_raw": data[24], "integral_limit_raw": data[25],
            "cw_deadband_raw": data[26], "ccw_deadband_raw": data[27],
            "current_limit_raw": word(28), "resolution_raw": data[30],
            "offset_register_raw": word(31), "mode": data[33],
            "overload_hold_percent": data[34], "overload_time_raw": data[35],
            "overload_output_percent": data[36], "speed_pid_p": data[37],
            "overcurrent_time_raw": data[38], "speed_pid_i": data[39],
            "packet_error": error, "registers_0_39_hex": data.hex(),
        }

    def ping(self, servo_id: int) -> int:
        """모터 응답을 확인하고 장치가 보고한 모델 번호를 반환한다."""
        return self._call(servo_id, "ping")[0]

    def read_register(self, servo_id: int, address: int) -> int:
        """초기 설정에 사용하는 한 바이트 레지스터를 읽는다."""
        _check_integer("레지스터 주소", address, 0, 255)
        return self._call(servo_id, "read1ByteTxRx", address)[0]

    def write_register(self, servo_id: int, address: int, value: int) -> None:
        """초기 설정에 사용하는 한 바이트 레지스터를 쓰고 응답을 확인한다."""
        _check_integer("레지스터 주소", address, 0, 255)
        _check_integer("레지스터 값", value, 0, 255)
        self._call(servo_id, "write1ByteTxRx", address, value)

    def read_position(self, servo_id: int) -> int:
        """모터의 현재 위치를 영점 변환하지 않은 엔코더 원시값으로 읽는다."""
        return self._call(servo_id, "ReadPos")[0]

    def read_position_speed(self, servo_id: int) -> tuple[int, int]:
        """현재 위치와 속도를 하나의 응답에서 원시값으로 읽는다."""
        position, speed = self._call(servo_id, "ReadPosSpeed")
        return position, speed

    def read_torque(self, servo_id: int) -> bool:
        """모터가 보고한 토크 켜짐 여부를 읽고 알 수 없는 상태는 오류로 알린다."""
        torque = self.read_register(servo_id, SMS_STS_TORQUE_ENABLE)
        if torque not in (0, 1):
            raise MotorError(f"모터 {servo_id}의 알 수 없는 토크 상태입니다: {torque}")
        return torque == 1

    def check_midpoint_setup(self, servo_id: int) -> int:
        """중점 설정 전에 단회전 모드와 토크 해제·정지 상태를 확인한다."""
        position, speed = self.read_position_speed(servo_id)
        _check_integer("현재 위치", position, 0, 4095)
        self._check_position_target(servo_id, MIDPOINT_RAW)
        if self.read_register(servo_id, 18) & 0x10:
            raise ValueError(f"모터 {servo_id}의 멀티턴 설정을 먼저 해제해 주세요.")
        if self.read_torque(servo_id):
            raise ValueError(f"모터 {servo_id}의 토크를 해제한 뒤 기준 자세에 맞춰 주세요.")
        if speed != 0:
            raise ValueError(f"모터 {servo_id}가 움직이고 있습니다. 기준 자세에서 멈춰 주세요.")
        return position

    def calibrate_midpoint(self, servo_id: int) -> int:
        """현재 자세를 모터 중점으로 재정의하고 읽기로 확인하며 토크는 해제 상태로 둔다.

        제조사의 중점 설정 명령만 보내며 위치 이동 명령은 보내지 않는다.
        응답이 불확실해도 중점 설정 명령을 자동으로 반복하지 않는다.
        """
        self.check_midpoint_setup(servo_id)
        try:
            self.write_register(servo_id, SMS_STS_TORQUE_ENABLE, 128)
        finally:
            self.write_register(servo_id, SMS_STS_TORQUE_ENABLE, 0)
        deadline = time.monotonic() + 0.5
        while True:
            position, speed = self.read_position_speed(servo_id)
            if position == MIDPOINT_RAW and speed == 0:
                if self.read_torque(servo_id):
                    raise MotorError(f"모터 {servo_id}의 토크 해제를 확인하지 못했습니다.")
                return position
            if time.monotonic() >= deadline:
                raise MotorError(f"모터 {servo_id}의 중점 설정을 확인하지 못했습니다. 현재 raw={position}")
            time.sleep(0.02)

    def _check_position_target(self, servo_id: int, position: int) -> None:
        """일반 위치 모드와 장치에 저장된 위치 범위를 확인하며 설정은 변경하지 않는다."""
        _check_integer("목표 위치", position, 0, 4095)
        mode = self._call(servo_id, "read1ByteTxRx", SMS_STS_MODE)[0]
        limits = self._call(servo_id, "read4ByteTxRx", SMS_STS_MIN_ANGLE_LIMIT_L)[0]
        lower = limits & 0xFFFF
        upper = limits >> 16
        if mode != 0 or (lower == 0 and upper == 0):
            raise ValueError("이번 모듈의 이동은 단회전 위치 모드만 지원합니다. 모터 설정은 변경하지 않았습니다.")
        if not lower <= position <= upper:
            raise ValueError(f"목표 {position}이 모터에 저장된 위치 범위 {lower}부터 {upper}를 벗어납니다.")

    def set_torque(self, servo_id: int, enabled: bool) -> None:
        """토크를 켜거나 끄며 켤 때는 현재 위치를 먼저 목표로 넣어 이전 목표로의 이동을 막는다."""
        if not isinstance(enabled, bool):
            raise ValueError("토크 상태에는 True 또는 False를 사용해 주세요.")
        if enabled:
            position = self.read_position(servo_id)
            self._check_position_target(servo_id, position)
            self._call(servo_id, "WritePosEx", position, 100, 10)
        self._call(servo_id, "write1ByteTxRx", SMS_STS_TORQUE_ENABLE, int(enabled))

    def move_to(
        self, servo_id: int, position: int, *, speed: int = 100, acceleration: int = 10,
    ) -> None:
        """목표 위치와 속도·가속도를 전송하고 필요하면 현재 위치에서 토크를 켠다.

        위치는 단회전 엔코더 원시값, 속도와 가속도는 모터 레지스터 단위다.
        전송 성공은 도착을 뜻하지 않으며 실제 위치는 읽기 함수로 확인한다.
        """
        _check_integer("속도", speed, 1, 3400)
        _check_integer("가속도", acceleration, 1, 254)
        self._check_position_target(servo_id, position)
        torque = self._call(servo_id, "read1ByteTxRx", SMS_STS_TORQUE_ENABLE)[0]
        if torque == 0:
            self.set_torque(servo_id, True)
        elif torque != 1:
            raise MotorError(f"모터 {servo_id}의 토크 상태 {torque}에서는 이동할 수 없습니다.")
        self._call(servo_id, "WritePosEx", position, speed, acceleration)

    def stop(self, servo_id: int) -> int:
        """현재 읽은 위치를 새 목표로 전송해 이동을 멈추며 토크 상태는 유지한다."""
        position = self.read_position(servo_id)
        self._check_position_target(servo_id, position)
        self._call(servo_id, "WritePosEx", position, 100, 10)
        return position
