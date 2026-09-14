"""이전 멀티턴 설정의 최대 위치와 Phase 비트만 단회전용으로 바꾼다.
모터 하나를 연결해 사용하며 ID·영점·목표 위치·토크는 유지한다.
"""

import argparse
import sys

from hardware.motor_ids import find_motor_id
from hardware.ports import resolve_port_settings
from hardware.sts3215 import MotorError, STS3215Bus


def _read_settings(bus: STS3215Bus, servo_id: int) -> dict[str, int]:
    """지정한 모터에서 변경 대상과 유지할 설정을 한 번에 읽는다."""
    raw = [0] * 5 + bus._call(servo_id, "readTxRx", 5, 51)[0]
    return {
        "id": raw[5],
        "minimum": raw[9] | raw[10] << 8,
        "maximum": raw[11] | raw[12] << 8,
        "phase": raw[18],
        "offset": raw[31] | raw[32] << 8,
        "mode": raw[33],
        "torque": raw[40],
        "lock": raw[55],
    }


def configure_single_turn(bus: STS3215Bus, servo_id: int) -> bool:
    """토크가 꺼진 모터의 최대 위치와 멀티턴 비트만 변경하고 저장값을 검증한다.

    이미 단회전이면 쓰지 않는다. 설정 도중 실패해도 설정 잠금을 복구한다.
    """
    before = _read_settings(bus, servo_id)
    if before["id"] != servo_id:
        raise MotorError("응답한 ID와 저장된 ID가 일치하지 않습니다.")
    if before["mode"] != 0 or before["minimum"] != 0:
        raise ValueError("위치 모드 0·최소 위치 0인 모터에만 적용합니다. 설정은 변경하지 않았습니다.")
    if before["maximum"] not in (0, 4095):
        raise ValueError("별도로 지정된 최대 위치가 있어 변경하지 않았습니다.")
    phase = before["phase"] & ~0x10
    if before["maximum"] == 4095 and before["phase"] == phase:
        return False
    if before["torque"] != 0:
        raise ValueError("토크를 끈 뒤 다시 실행해 주세요. 설정은 변경하지 않았습니다.")

    try:
        bus.write_register(servo_id, 55, 0)
        if bus.read_register(servo_id, 55) != 0:
            raise MotorError("설정 잠금을 해제하지 못했습니다.")
        if before["maximum"] != 4095:
            bus._call(servo_id, "write2ByteTxRx", 11, 4095)
        if before["phase"] != phase:
            bus.write_register(servo_id, 18, phase)
        expected = before | {"maximum": 4095, "phase": phase, "lock": 0}
        if _read_settings(bus, servo_id) != expected:
            raise MotorError("설정 읽기 검증에 실패했습니다. 일부 설정이 변경되었을 수 있습니다.")
    finally:
        bus.write_register(servo_id, 55, 1)
        if bus.read_register(servo_id, 55) != 1:
            raise MotorError("설정 잠금을 확인하지 못했습니다. 연결 상태를 확인하고 다시 실행해 주세요.")
    return True


def main(argv: list[str] | None = None) -> int:
    """저장된 포트의 모터 하나를 현재 제어 코드에 맞는 단회전 설정으로 준비한다."""
    parser = argparse.ArgumentParser(description="STS3215 단회전 초기 설정: 모터 하나만 연결")
    parser.add_argument("--id", type=int, help="지정하면 해당 ID만 조회, 생략하면 자동 검색")
    parser.add_argument("--port", help="생략하면 저장된 포트 사용")
    parser.add_argument("--baudrate", type=int, help="생략하면 선택한 포트의 통신 속도 사용")
    args = parser.parse_args(argv)
    try:
        settings = resolve_port_settings(args.port, args.baudrate)
        with STS3215Bus(settings.port, settings.baudrate) as bus:
            servo_id = args.id if args.id is not None else find_motor_id(bus)
            print(f"사용 포트: {settings.port}, 모터 ID: {servo_id}", flush=True)
            changed = configure_single_turn(bus, servo_id)
        result = "단회전 설정 완료" if changed else "이미 단회전 설정입니다"
        print(f"ID {servo_id}: {result}. ID·영점·토크는 유지했습니다.")
    except KeyboardInterrupt:
        print("설정을 중단했습니다. 다시 실행하면 현재 저장값부터 확인합니다.", file=sys.stderr)
        return 130
    except (MotorError, OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
