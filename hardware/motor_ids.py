"""연결된 모터의 현재 ID를 찾고 입력받은 새 ID로 변경한 뒤 확인한다.
초기 설정 때 모터 하나만 연결해서 사용하며 관절 보정 파일은 수정하지 않는다.
"""

import argparse
import sys
import time

from scservo_sdk import COMM_RX_TIMEOUT
from scservo_sdk.sms_sts import SMS_STS_ID, SMS_STS_LOCK

from hardware.ports import resolve_port_settings
from hardware.sts3215 import MotorError, STS3215Bus


def find_motor_id(bus: STS3215Bus) -> int:
    """연결한 포트에서 ID를 순서대로 읽고 처음 응답한 모터의 ID를 반환한다.

    모터 하나만 연결한 초기 설정용이며 여러 모터의 개수를 판별하는 검색은 아니다.
    기본 ID부터 검색하고 응답이 확인되면 나머지 ID는 조회하지 않는다.
    """
    for servo_id in (*range(1, 254), 0):
        try:
            stored_id = bus.read_register(servo_id, SMS_STS_ID)
        except MotorError as error:
            if error.communication_result == COMM_RX_TIMEOUT:
                continue
            raise
        if stored_id != servo_id:
            raise MotorError(f"응답한 ID {servo_id}와 저장된 ID {stored_id}가 일치하지 않습니다.")
        return servo_id
    raise MotorError("응답하는 모터를 찾지 못했습니다. 모터 전원·연결·통신 속도를 확인해 주세요.")


def _prompt_new_id() -> int | None:
    """목표 ID를 입력받고 빈 입력이나 입력 종료는 변경 취소로 처리한다."""
    while True:
        try:
            entered = input("변경할 ID를 입력하세요 (0~253, Enter: 취소): ").strip()
        except EOFError:
            return None
        if not entered:
            return None
        try:
            new_id = int(entered)
        except ValueError:
            print("0부터 253 사이의 정수를 입력해 주세요.")
            continue
        if 0 <= new_id <= 253:
            return new_id
        print("0부터 253 사이의 정수를 입력해 주세요.")


def _relock_after_failure(bus: STS3215Bus, current_id: int, new_id: int) -> str:
    """설정 도중 실패하면 새 ID와 기존 ID 중 응답하는 모터의 설정 잠금을 복구한다."""
    responding_ids = []
    for servo_id in (new_id, current_id):
        try:
            bus.ping(servo_id)
            responding_ids.append(servo_id)
            bus.write_register(servo_id, SMS_STS_LOCK, 1)
            if bus.read_register(servo_id, SMS_STS_LOCK) == 1:
                return f"ID {servo_id} 응답과 설정 잠금을 확인했습니다."
        except (MotorError, OSError):
            continue
    if responding_ids:
        return f"응답한 ID는 {responding_ids}이며 설정 잠금을 확인하지 못했습니다. 이 ID의 설정 잠금을 확인해 주세요."
    return f"ID {current_id} 또는 {new_id}의 설정 잠금을 확인하지 못했습니다. 연결 후 ID와 잠금을 확인해 주세요."


def change_motor_id(bus: STS3215Bus, current_id: int, new_id: int) -> int:
    """모터 하나의 ID만 변경하고 새 ID의 응답·모델 번호·설정 잠금을 검증한다.

    통신 보드에는 모터 하나만 연결해야 한다. 같은 ID의 모터 여러 개는 구별할 수 없다.
    위치·토크·통신 속도·영점은 바꾸지 않는다. 실패 시 ID 쓰기를 자동 반복하지 않는다.
    """
    for name, value in (("현재 ID", current_id), ("새 ID", new_id)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 253:
            raise ValueError(f"{name}는 0부터 253 사이의 정수여야 합니다.")
    model = bus.ping(current_id)
    if bus.read_register(current_id, SMS_STS_ID) != current_id:
        raise MotorError("현재 ID의 응답과 저장된 ID가 일치하지 않습니다.")
    if current_id == new_id:
        return new_id
    try:
        bus.ping(new_id)
    except MotorError as error:
        if error.communication_result != COMM_RX_TIMEOUT:
            raise
    else:
        raise ValueError(f"새 ID {new_id}에서 이미 응답합니다. 모터 하나만 연결하고 사용하지 않는 ID를 지정해 주세요.")

    try:
        bus.write_register(current_id, SMS_STS_LOCK, 0)
        if bus.read_register(current_id, SMS_STS_LOCK) != 0:
            raise MotorError("모터 설정 잠금을 해제하지 못했습니다.")
        try:
            bus.write_register(current_id, SMS_STS_ID, new_id)
        except MotorError as error:
            if error.communication_result != COMM_RX_TIMEOUT:
                raise
        time.sleep(0.05)
        if bus.ping(new_id) != model or bus.read_register(new_id, SMS_STS_ID) != new_id:
            raise MotorError("새 ID의 모델 번호 또는 저장된 ID가 예상과 다릅니다.")
        bus.write_register(new_id, SMS_STS_LOCK, 1)
        if bus.read_register(new_id, SMS_STS_LOCK) != 1:
            raise MotorError("새 ID에서 설정 잠금을 확인하지 못했습니다.")
    except (MotorError, OSError, KeyboardInterrupt) as error:
        recovery = _relock_after_failure(bus, current_id, new_id)
        if isinstance(error, KeyboardInterrupt):
            print(recovery, file=sys.stderr)
            raise
        raise MotorError(f"ID 설정을 완료하지 못했습니다: {error} {recovery}") from error
    return new_id


def main(argv: list[str] | None = None) -> int:
    """현재 모터 ID를 확인해 출력하고 목표 ID를 입력받아 변경한다."""
    parser = argparse.ArgumentParser(
        description="STS3215 모터 ID 초기 설정",
        epilog="ID를 설정할 모터 하나만 통신 보드에 연결해 주세요.",
    )
    parser.add_argument("--current-id", type=int, help="생략하면 연결된 모터의 ID를 자동 검색")
    parser.add_argument("--new-id", type=int, help="생략하면 현재 ID를 출력한 뒤 목표 ID 입력")
    parser.add_argument("--port", help="생략하면 저장된 포트 사용")
    parser.add_argument("--baudrate", type=int, help="생략하면 선택한 포트의 통신 속도 사용")
    args = parser.parse_args(argv)
    try:
        settings = resolve_port_settings(args.port, args.baudrate)
        print(f"사용 포트: {settings.port}. ID를 설정할 모터 하나만 연결한 상태에서 사용합니다.", flush=True)
        with STS3215Bus(settings.port, settings.baudrate) as bus:
            if args.current_id is None:
                print("연결된 모터의 현재 ID를 찾는 중입니다...", flush=True)
                current_id = find_motor_id(bus)
            else:
                current_id = args.current_id
                if bus.read_register(current_id, SMS_STS_ID) != current_id:
                    raise MotorError("현재 ID의 응답과 저장된 ID가 일치하지 않습니다.")
            print(f"현재 모터 ID: {current_id}", flush=True)
            new_id = args.new_id if args.new_id is not None else _prompt_new_id()
            if new_id is None:
                print("ID 변경을 취소했습니다.")
                return 0
            change_motor_id(bus, current_id, new_id)
        print(f"ID 확인 완료: {current_id} → {new_id}")
    except KeyboardInterrupt:
        print("ID 설정을 중단했습니다.", file=sys.stderr)
        return 130
    except (MotorError, OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
