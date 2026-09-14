"""직렬 포트 목록을 표시하고 선택한 포트와 통신 속도를 로컬에 저장한다.
저장값을 사용할 때는 전체 포트를 검색하거나 다른 포트로 자동 전환하지 않는다.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from serial.tools.list_ports import comports
from serial.tools.list_ports_common import ListPortInfo


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / ".local" / "motor_port.json"
DEFAULT_BAUDRATE = 1_000_000


@dataclass(frozen=True)
class PortSettings:
    """모터 통신에 사용할 포트 경로와 통신 속도를 담는다."""

    port: str
    baudrate: int = DEFAULT_BAUDRATE

    def __post_init__(self) -> None:
        """비어 있는 포트와 잘못된 통신 속도를 저장하기 전에 거부한다."""
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("포트 경로를 입력해 주세요.")
        if isinstance(self.baudrate, bool) or not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("통신 속도는 양의 정수여야 합니다.")


def list_ports() -> list[ListPortInfo]:
    """운영체제의 포트 목록만 읽으며 모터에 통신 명령은 보내지 않는다."""
    return sorted(comports(), key=lambda item: item.device)


def load_port_settings(path: Path | None = None) -> PortSettings | None:
    """저장된 연결 설정을 읽고 파일이 없으면 선택되지 않은 상태를 반환한다."""
    source = DEFAULT_SETTINGS_PATH if path is None else Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"포트 설정 파일을 읽을 수 없습니다. ports save로 다시 저장해 주세요: {source}") from error
    if not isinstance(payload, dict) or set(payload) != {"port", "baudrate"}:
        raise ValueError(f"포트 설정 파일의 형식이 잘못되었습니다: {source}")
    return PortSettings(**payload)


def save_port_settings(settings: PortSettings, path: Path | None = None) -> Path:
    """연결 설정을 임시 파일에 쓴 뒤 교체하며 실제 포트 연결은 시도하지 않는다."""
    destination = DEFAULT_SETTINGS_PATH if path is None else Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(asdict(settings), temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def resolve_port_settings(port: str | None = None, baudrate: int | None = None) -> PortSettings:
    """명시한 포트 또는 저장한 포트를 선택하며 전체 포트 목록을 조회하지 않는다."""
    if port is not None:
        return PortSettings(port, DEFAULT_BAUDRATE if baudrate is None else baudrate)
    settings = load_port_settings()
    if settings is None:
        raise ValueError("저장된 포트가 없습니다. python -m hardware.ports list로 찾고 save --port로 저장해 주세요.")
    return PortSettings(settings.port, settings.baudrate if baudrate is None else baudrate)


def main(argv: list[str] | None = None) -> int:
    """포트 목록 표시·선택 저장·저장값 표시 중 하나를 실행한다."""
    parser = argparse.ArgumentParser(description="모터 통신 포트 찾기와 선택 저장")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="사용 가능한 포트 목록 표시")
    subparsers.add_parser("show", help="저장된 포트만 표시")
    save_parser = subparsers.add_parser("save", help="선택한 포트와 통신 속도 저장")
    save_parser.add_argument("--port", required=True, help="선택한 포트 경로")
    save_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            ports = list_ports()
            for port in ports:
                print(f"{port.device}  {port.description}")
            if not ports:
                print("사용 가능한 직렬 통신 포트가 없습니다.")
        elif args.action == "save":
            settings = PortSettings(args.port, args.baudrate)
            save_port_settings(settings)
            print(f"포트 저장: {settings.port}, 통신 속도={settings.baudrate}. 실제 연결은 읽기 명령에서 확인합니다.")
        else:
            settings = resolve_port_settings()
            print(f"저장된 포트: {settings.port}, 통신 속도={settings.baudrate}")
    except (OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
