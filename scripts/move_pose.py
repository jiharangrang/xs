"""GUI에서 복사한 도 단위 관절각을 기존 서버로 보내 한 번 이동하고 raw 오차를 출력한다.
직렬 포트는 서버가 계속 소유하며, 그리퍼는 명시적으로 포함할 때만 움직인다.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from kinematics.joints import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES


class PoseInterrupted(ValueError):
    """다른 명령으로 대체된 자세를 더 이상 관찰하지 않도록 알린다."""


def parse_angles(text: str, *, include_grippers: bool = False) -> dict[str, float]:
    """복사한 JSON의 관절 이름과 도 단위 값을 확인하고 실제 이동 대상을 고른다."""
    angles = json.loads(text)
    if not isinstance(angles, dict):
        raise ValueError('관절 이름별 JSON을 넣어 주세요. 예: {"J1": 0, ...}')
    if set(angles) - set(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES):
        raise ValueError("관절 이름에는 J1~J7, G_L, G_R만 사용할 수 있습니다.")
    missing = set(ARM_JOINT_NAMES) - angles.keys()
    if missing:
        raise ValueError("몸통 관절각이 빠졌습니다: " + ", ".join(sorted(missing)))
    for joint, angle in angles.items():
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle):
            raise ValueError(f"{joint}: 각도에는 유한한 숫자를 넣어 주세요.")
    names = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAMES if include_grippers else ())
    return {name: float(angles[name]) for name in names if name in angles}


def post(server: str, route: str, body: dict) -> dict:
    """모터 포트를 다시 열지 않고 실행 중인 서버에 명령을 한 번 전송한다."""
    request = Request(server.rstrip("/") + route, json.dumps(body, allow_nan=False).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        payload = json.loads(error.read())
        raise ValueError(f"서버가 명령을 거부했습니다: {payload.get('detail', payload)}") from error


def observe_pose(server: str, command_ids: dict[str, int], *, timeout: float = 45.0) -> list[dict]:
    """이번 명령에 해당하는 상태만 모아 도착 또는 미도착 판정까지 조용히 기다린다."""
    url = server.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws"
    deadline = time.monotonic() + timeout
    with connect(url, open_timeout=5, close_timeout=1) as socket:
        while time.monotonic() < deadline:
            payload = json.loads(socket.recv(timeout=min(5.0, max(0.01, deadline - time.monotonic()))))
            states = {state["name"]: state for state in payload["joints"]}
            selected = []
            for joint, command_id in command_ids.items():
                state = states.get(joint)
                if state is None:
                    raise ValueError(f"{joint}: 상태가 누락되었습니다.")
                if state.get("error"):
                    raise ValueError(f"{joint}: {state['error']}")
                received_id = state.get("command_id")
                if received_id is None or received_id < command_id:
                    continue
                if received_id != command_id or state["motion_status"] == "idle":
                    raise PoseInterrupted(f"{joint}: 다른 명령 또는 토크 해제로 이번 이동이 중단되었습니다.")
                selected.append(state)
            if len(selected) == len(command_ids):
                if all(state["motion_status"] in ("arrived", "timeout") for state in selected):
                    return selected
    raise TimeoutError("제한 시간 안에 이동 결과를 받지 못했습니다.")


def print_result(states: list[dict]) -> bool:
    """도착 관절 수와 관절별 절대 raw 오차를 두 줄로 출력한다."""
    arrived = sum(state["motion_status"] == "arrived" for state in states)
    outside = [state["name"] for state in states if abs(state["error_raw"]) > state["tolerance_raw"]]
    limits = {state["tolerance_raw"] for state in states}
    criterion = f"{next(iter(limits)):g} raw 이내·정지" if len(limits) == 1 else "관절별 공통 기준"
    print(f"도착 {arrived}/{len(states)} ({criterion}) | 허용 초과: {', '.join(outside) or '없음'}")
    print("오차(raw): " + "  ".join(f"{state['name']}={abs(state['error_raw'])}" for state in states))
    return arrived == len(states)


def stop_targets(server: str, joints: dict) -> None:
    """관찰 중단 시 대상 관절만 현재 위치에서 멈추고 토크는 유지한다."""
    failed = []
    for joint in joints:
        try:
            post(server, "/api/stop", {"joint": joint})
        except (OSError, ValueError):
            failed.append(joint)
    message = "정지 확인 실패: " + ", ".join(failed) if failed else "대상 관절 정지 명령 전송. 토크 유지."
    print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """JSON을 입력받아 몸통 관절을 묶음 이동시키고 간단한 결과만 표시한다."""
    parser = argparse.ArgumentParser(description="관절각 JSON으로 한 번 자세 이동", allow_abbrev=False)
    parser.add_argument("angles", nargs="?", help="GUI에서 복사한 도 단위 JSON, 생략하면 붙여넣기 입력")
    parser.add_argument("--file", type=Path, help="관절각 JSON 파일")
    parser.add_argument("--server", default="http://127.0.0.1:18765", help="실행 중인 캘리브레이션 서버")
    parser.add_argument("--include-grippers", action="store_true", help="JSON에 적힌 그리퍼 각도도 이동")
    args = parser.parse_args(argv)
    if args.angles is not None and args.file is not None:
        parser.error("JSON 직접 입력과 --file 중 하나만 사용해 주세요.")
    if not args.server.startswith(("http://", "https://")):
        parser.error("--server에는 http:// 또는 https:// 주소를 넣어 주세요.")
    accepted = None
    requested = None
    try:
        if args.file is not None:
            text = args.file.read_text(encoding="utf-8")
        elif args.angles is not None:
            text = args.angles
        else:
            text = input("관절각 JSON(도)을 붙여넣으세요: ") if sys.stdin.isatty() else sys.stdin.read()
        angles = parse_angles(text, include_grippers=args.include_grippers)
        requested = angles
        accepted = post(args.server, "/api/pose", {"angles_deg": angles})
        grippers = "입력한 그리퍼 포함" if args.include_grippers else "그리퍼 유지"
        print(f"{len(angles)}관절 목표 전송 · {grippers}", flush=True)
        result = print_result(observe_pose(args.server, accepted["command_ids"]))
        print(f"로그: {accepted['log_path']}")
        return 0 if result else 1
    except KeyboardInterrupt:
        if requested is not None:
            stop_targets(args.server, requested)
        return 130
    except PoseInterrupted as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, URLError, WebSocketException, EOFError) as error:
        print(f"오류: {error}", file=sys.stderr)
        if accepted is not None:
            stop_targets(args.server, accepted["command_ids"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
