"""STS3215 모터 하나의 위치 읽기와 작은 이동을 터미널에서 실행한다.
모터의 원시 위치값을 사용하며 기본 명령은 읽기다.
"""

import argparse
import math
import sys
import time

from hardware.ports import list_ports, resolve_port_settings
from hardware.sts3215 import MotorError, STS3215Bus


def _observe_move(bus: STS3215Bus, servo_id: int, target: int, timeout: float) -> bool:
    """현재 위치와 속도를 출력하며 목표 근처에서 정지했는지 제한 시간 동안 확인한다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        position, speed = bus.read_position_speed(servo_id)
        print(f"현재 위치={position}, 목표={target}, 현재 속도={speed}", flush=True)
        if abs(position - target) <= 10 and speed == 0:
            print("목표와의 차이가 10카운트 이내이고 속도가 0입니다.")
            return True
        time.sleep(0.1)
    return False


def _try_stop(bus: STS3215Bus, servo_id: int) -> None:
    """중단 시 현재 위치 유지 명령을 시도하고 실패하면 정지 여부를 알 수 없음을 표시한다."""
    try:
        position = bus.stop(servo_id)
        print(f"정지 명령 전송: 목표 위치={position}. 토크 상태는 유지합니다.", file=sys.stderr)
    except (MotorError, OSError, ValueError) as error:
        print(f"정지 명령 실패. 실제 정지 여부를 확인해 주세요: {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """입력한 포트와 모터 ID에 읽기·이동·토크·정지 명령 중 하나를 실행한다."""
    parser = argparse.ArgumentParser(description="STS3215 모터 하나 읽기·이동")
    parser.add_argument("action", nargs="?", default="read", choices=["read", "move", "torque-on", "torque-off", "stop"])
    parser.add_argument("--list-ports", action="store_true", help="사용 가능한 포트만 표시")
    parser.add_argument("--port", help="통신 보드의 포트 경로, 생략하면 저장된 포트 사용")
    parser.add_argument("--id", type=int, help="모터 ID")
    parser.add_argument("--baudrate", type=int, help="생략하면 선택한 포트의 통신 속도 사용")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--delta", type=int, help="현재 위치에서 이동할 카운트, move에서 사용")
    target.add_argument("--position", type=int, help="목표 원시 위치, move에서 사용")
    parser.add_argument("--speed", type=int, default=100, help="이동 속도 원시값, 기본 100")
    parser.add_argument("--acceleration", type=int, default=10, help="가속도 원시값, 기본 10")
    parser.add_argument("--timeout", type=float, default=5.0, help="이동 관찰 제한 시간(초), 기본 5")
    args = parser.parse_args(argv)
    if args.list_ports:
        ports = list_ports()
        for port in ports:
            print(f"{port.device}  {port.description}")
        if not ports:
            print("사용 가능한 직렬 통신 포트가 없습니다.")
        return 0
    if args.id is None:
        parser.error("--id로 모터 ID를 지정해 주세요.")
    if args.action == "move" and args.delta is None and args.position is None:
        parser.error("move에는 --delta 또는 --position을 지정해 주세요.")
    if args.action != "move" and (args.delta is not None or args.position is not None):
        parser.error("--delta와 --position은 move에서만 사용합니다.")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout은 유한한 양수여야 합니다.")

    try:
        settings = resolve_port_settings(args.port, args.baudrate)
        with STS3215Bus(settings.port, settings.baudrate) as bus:
            if args.action == "read":
                print(f"모터 ID={args.id}, 모델 번호={bus.ping(args.id)}")
                position, speed = bus.read_position_speed(args.id)
                print(f"현재 위치={position}, 현재 속도={speed} (모터 원시값)")
            elif args.action in ("torque-on", "torque-off"):
                enabled = args.action == "torque-on"
                bus.set_torque(args.id, enabled)
                print("현재 위치에서 토크 켜기 명령을 전송했습니다." if enabled else "토크 해제 명령을 전송했습니다.")
            elif args.action == "stop":
                print(f"정지 명령 전송: 목표 위치={bus.stop(args.id)}")
            else:
                current = bus.read_position(args.id)
                position = args.position if args.position is not None else current + args.delta
                print(f"모터 ID={args.id}, 시작={current}, 목표={position}, 속도={args.speed}", flush=True)
                try:
                    bus.move_to(args.id, position, speed=args.speed, acceleration=args.acceleration)
                    if not _observe_move(bus, args.id, position, args.timeout):
                        raise MotorError("제한 시간 안에 목표 도착을 확인하지 못했습니다.")
                except (MotorError, OSError, KeyboardInterrupt):
                    _try_stop(bus, args.id)
                    raise
                print("관찰 종료. 모터는 목표 위치에서 토크를 유지합니다.")
    except KeyboardInterrupt:
        print("사용자가 명령을 중단했습니다.", file=sys.stderr)
        return 130
    except (MotorError, OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
