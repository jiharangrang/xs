"""모터 명령과 관측값을 실행별 JSONL 파일 하나에 기록한다.
통신이나 제어는 수행하지 않으며, 저장 실패가 모터 명령을 중단하지 않게 한다.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "motors"


class MotorLogger:
    """공통 제어 경로에서 전달한 명령·설정·피드백을 시간순으로 저장한다."""

    def __init__(self, directory: str | Path | None = None, *, enabled: bool = True) -> None:
        """파일 이름만 준비하고 첫 기록이 들어올 때 파일을 생성한다."""
        directory = DEFAULT_LOG_DIR if directory is None else Path(directory)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = directory / f"{stamp}_{uuid4().hex[:8]}.jsonl"
        self.enabled = enabled
        self.context: dict = {}
        self._created = False
        self._sequence = 0
        self._command_ids: dict[str, int] = {}

    def write(self, event: str, **fields) -> None:
        """한 줄씩 즉시 저장하고 실패하면 한 번 알린 뒤 로깅만 중단한다."""
        if not self.enabled:
            return
        record = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(),
                  "monotonic_s": time.monotonic(), **fields}
        try:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n"
            if not self._created:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("x", encoding="utf-8") as stream:
                    header = {"event": "session", "schema_version": 1,
                              "time_utc": record["time_utc"], "monotonic_s": record["monotonic_s"],
                              **self.context}
                    stream.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")
                self._created = True
                print(f"모터 로그: {self.path}", file=sys.stderr, flush=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except (OSError, ValueError, TypeError) as error:
            self.enabled = False
            print(f"모터 로그 저장 실패. 제어는 계속하며 로깅만 중단합니다: {error}", file=sys.stderr, flush=True)

    def command_id(self, joint: str) -> int | None:
        """해당 관절에 가장 최근 요청한 명령 번호를 반환한다."""
        return self._command_ids.get(joint)

    @contextmanager
    def command(self, joint: str, action: str, **parameters):
        """명령 요청과 전송 결과를 같은 번호로 기록하며 기존 예외를 그대로 전달한다."""
        self._sequence += 1
        command_id = self._sequence
        self._command_ids[joint] = command_id
        self.write("command", joint=joint, command_id=command_id, action=action, **parameters)
        result = {"joint": joint, "command_id": command_id, "action": action}
        try:
            yield result
        except BaseException as error:
            self.write("command_result", **result, result="failed", error=str(error),
                       error_type=type(error).__name__,
                       communication_result=getattr(error, "communication_result", None),
                       device_error=getattr(error, "device_error", None))
            raise
        else:
            self.write("command_result", **result, result="sent")
