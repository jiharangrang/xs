"""공통 관절 제어 모듈을 통해 도 단위 읽기·이동·영점 저장을 실행한다.
위치·속도·가속도·도착 오차는 사람이 사용하는 각도 단위로 입력하고 출력한다.
"""

import argparse
import math
import sys
import time

from hardware.calibration import MotorCalibration
from hardware.joint_control import DEFAULT_CALIBRATION_PATH, JointController
from hardware.ports import list_ports, resolve_port_settings
from hardware.sts3215 import MotorError, STS3215Bus


def _observe_move(
    controller: JointController, joint: str, target_deg: float, timeout: float,
) -> bool:
    """현재 각도와 속도를 출력하며 지정한 각도 오차 안에서 정지했는지 확인한다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = controller.read(joint)
        print(f"{joint}: 현재={state.position_deg:.3f}°, 목표={target_deg:.3f}°, 속도={state.speed_deg_s:.3f}°/s", flush=True)
        if controller.has_arrived(state, target_deg):
            print(f"목표와의 차이가 {controller.tolerance_for(joint):g}° 이내이고 속도가 0°/s입니다.")
            return True
        time.sleep(0.1)
    return False


def _try_stop(controller: JointController, joint: str) -> None:
    """중단 시 현재 각도 유지 명령을 시도하고 실패하면 정지 여부를 알 수 없음을 표시한다."""
    try:
        angle = controller.stop(joint)
        print(f"정지 명령 전송: {joint} 목표={angle:.3f}°. 토크 상태는 유지합니다.", file=sys.stderr)
    except (MotorError, OSError, ValueError) as error:
        print(f"정지 명령 실패. 실제 정지 여부를 확인해 주세요: {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """관절 이름 또는 기존 ID 입력을 공통 도 단위 제어 통로에 전달한다."""
    parser = argparse.ArgumentParser(description="STS3215 관절을 도 단위로 읽기·이동·영점 저장", allow_abbrev=False)
    parser.add_argument("action", nargs="?", default="read", choices=["read", "move", "zero", "torque-on", "torque-off", "stop"])
    parser.add_argument("--list-ports", action="store_true", help="사용 가능한 포트만 표시")
    parser.add_argument("--port", help="생략하면 저장된 포트 사용")
    parser.add_argument("--baudrate", type=int, help="생략하면 선택한 포트의 통신 속도 사용")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--joint", help="관절 이름: J1~J7, G_L, G_R")
    selector.add_argument("--id", type=int, help="설정 파일에서 해당 ID의 관절을 선택")
    selector.add_argument("--all", action="store_true", help="read에서 등록된 관절 전체 읽기")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH), help="관절 영점·방향 설정 파일")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--delta-deg", type=float, help="현재 각도에서 이동할 각도(°)")
    target.add_argument("--angle-deg", type=float, help="영점 기준 목표 각도(°)")
    parser.add_argument("--speed-deg-s", type=float, default=10.0, help="속도 크기(°/s), 기본 10")
    parser.add_argument("--acceleration-deg-s2", type=float, default=90.0, help="가속도 크기(°/s²), 기본 90")
    parser.add_argument("--tolerance-deg", type=float, help="도착 판정 오차(°), 생략하면 공통 모듈 기본값")
    parser.add_argument("--timeout", type=float, default=5.0, help="이동 관찰 제한 시간(초), 기본 5")
    args = parser.parse_args(argv)
    if args.list_ports:
        ports = list_ports()
        for port in ports:
            print(f"{port.device}  {port.description}")
        if not ports:
            print("사용 가능한 직렬 통신 포트가 없습니다.")
        return 0
    if args.joint is None and args.id is None and not args.all:
        parser.error("--joint 또는 --id로 관절을 선택하거나 read --all을 사용해 주세요.")
    if args.all and args.action != "read":
        parser.error("--all은 read에서만 사용합니다.")
    if args.action == "move" and args.delta_deg is None and args.angle_deg is None:
        parser.error("move에는 --delta-deg 또는 --angle-deg를 지정해 주세요.")
    if args.action != "move" and (args.delta_deg is not None or args.angle_deg is not None):
        parser.error("목표 각도는 move에서만 사용합니다.")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout은 유한한 양수여야 합니다.")

    try:
        settings = resolve_port_settings(args.port, args.baudrate)
        with STS3215Bus(settings.port, settings.baudrate) as bus:
            controller = JointController(bus, args.calibration, tolerance_deg=args.tolerance_deg)
            joint = args.joint if args.id is None else controller.joint_for_id(args.id)
            if args.action == "read":
                states = controller.read_all() if args.all else [controller.read(joint)]
                for state in states:
                    print(f"{state.name}: 현재 각도={state.position_deg:.3f}°, 현재 속도={state.speed_deg_s:.3f}°/s")
            elif args.action == "zero":
                MotorCalibration(bus, args.calibration).save_zero(joint)
                print(f"{joint}: 현재 기준 자세를 0°로 저장했습니다. 모터 설정은 변경하지 않았습니다.")
            elif args.action in ("torque-on", "torque-off"):
                enabled = args.action == "torque-on"
                controller.set_torque(joint, enabled)
                print(f"{joint}: 현재 위치에서 토크 켜기" if enabled else f"{joint}: 토크 해제")
            elif args.action == "stop":
                print(f"정지 명령 전송: {joint} 목표={controller.stop(joint):.3f}°")
            else:
                try:
                    options = {"speed_deg_s": args.speed_deg_s, "acceleration_deg_s2": args.acceleration_deg_s2}
                    if args.delta_deg is None:
                        target_deg = controller.move_to(joint, args.angle_deg, **options)
                    else:
                        target_deg = controller.move_by(joint, args.delta_deg, **options)
                    print(f"{joint}: 목표={target_deg:.3f}°, 지정 속도={args.speed_deg_s:g}°/s", flush=True)
                    if not _observe_move(controller, joint, target_deg, args.timeout):
                        raise MotorError("제한 시간 안에 목표 도착을 확인하지 못했습니다.")
                except (MotorError, OSError, KeyboardInterrupt):
                    _try_stop(controller, joint)
                    raise
                print("관찰 종료. 모터는 목표 각도에서 토크를 유지합니다.")
    except KeyboardInterrupt:
        print("사용자가 명령을 중단했습니다.", file=sys.stderr)
        return 130
    except (MotorError, OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
