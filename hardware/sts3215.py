"""제조사 SDK로 STS3215 모터의 위치를 읽고 단회전 위치 명령을 보낸다.
관절 영점 변환 없이 모터 원시값을 사용하며, 한 통신 포트는 한 실행 흐름에서 사용한다.
"""

from typing import Self

from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import (
    SMS_STS_MIN_ANGLE_LIMIT_L,
    SMS_STS_MODE,
    SMS_STS_TORQUE_ENABLE,
)


class MotorError(RuntimeError):
    """모터의 통신 실패 또는 장치 오류를 나타낸다."""

    def __init__(self, message: str, *, communication_result: int | None = None) -> None:
        """오류 설명과 SDK 통신 결과를 보관해 무응답과 다른 실패를 구분한다."""
        super().__init__(message)
        self.communication_result = communication_result


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

    def _call(self, servo_id: int, method: str, *args: int) -> list[int]:
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
        if error:
            raise MotorError(
                f"모터 {servo_id} 장치 오류 0x{error:02x}: {self._sdk.getRxPacketError(error)}"
            )
        return values

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
