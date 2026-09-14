"""포트 저장·재사용과 명시적 포트 선택이 전체 검색 없이 동작하는지 확인한다."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware.ports import (
    PortSettings, load_port_settings, main, resolve_port_settings, save_port_settings,
)
from scripts.test_motor import main as test_motor_main


class PortSettingsTests(unittest.TestCase):
    """임시 설정 파일을 사용해 실제 컴퓨터의 포트 선택을 바꾸지 않고 검증한다."""

    def setUp(self) -> None:
        """각 시험의 연결 설정을 별도 임시 폴더에 저장한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "local" / "motor_port.json"
        config_patch = patch("hardware.ports.DEFAULT_SETTINGS_PATH", self.path)
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def test_saved_port_is_reused_without_listing(self) -> None:
        """선택한 포트와 속도를 저장한 뒤 검색 없이 그대로 사용한다."""
        settings = PortSettings("/dev/cu.motor-example", 115200)
        with patch("hardware.ports.comports", side_effect=AssertionError("전체 포트 검색 금지")):
            save_port_settings(settings)
            self.assertEqual(load_port_settings(), settings)
            self.assertEqual(resolve_port_settings(), settings)
            self.assertEqual(resolve_port_settings(baudrate=500000), PortSettings(settings.port, 500000))
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_explicit_port_bypasses_saved_settings(self) -> None:
        """포트를 직접 지정하면 저장 파일과 전체 포트 목록을 읽지 않는다."""
        with patch("hardware.ports.load_port_settings", side_effect=AssertionError("설정 조회 금지")):
            with patch("hardware.ports.comports", side_effect=AssertionError("검색 금지")):
                self.assertEqual(resolve_port_settings("COM4"), PortSettings("COM4"))

    def test_missing_or_corrupt_settings_do_not_auto_select(self) -> None:
        """설정이 없거나 손상되어도 다른 포트를 자동 선택하지 않는다."""
        with patch("hardware.ports.comports", side_effect=AssertionError("검색 금지")):
            with self.assertRaisesRegex(ValueError, "저장된 포트가 없습니다"):
                resolve_port_settings()
            self.path.parent.mkdir(parents=True)
            for payload in ('{', '[]', '{"port": "COM4", "baudrate": true}', '{"port": "", "baudrate": 1000000}'):
                with self.subTest(payload=payload):
                    self.path.write_text(payload)
                    with self.assertRaises(ValueError):
                        resolve_port_settings()

    def test_cli_save_and_show_do_not_connect_or_list(self) -> None:
        """저장과 저장값 표시는 통신 포트를 열거나 전체 목록을 검색하지 않는다."""
        output = StringIO()
        with patch("hardware.ports.comports", side_effect=AssertionError("검색 금지")):
            with patch("serial.Serial", side_effect=AssertionError("연결 금지")), redirect_stdout(output):
                self.assertEqual(main(["save", "--port", "COM4", "--baudrate", "115200"]), 0)
                self.assertEqual(main(["show"]), 0)
        self.assertIn("COM4", output.getvalue())
        self.assertIn("115200", output.getvalue())

    def test_read_opens_saved_port_and_does_not_fallback(self) -> None:
        """읽기 실행은 저장한 포트만 열며 실패해도 다른 포트를 찾지 않는다."""
        save_port_settings(PortSettings("COM4", 115200))
        with patch("hardware.ports.comports", side_effect=AssertionError("검색 금지")):
            with patch("serial.Serial", side_effect=OSError("저장된 장치 없음")) as serial_factory:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(test_motor_main(["read", "--id", "1"]), 1)
        self.assertEqual(serial_factory.call_count, 1)
        self.assertEqual(serial_factory.call_args.kwargs["port"], "COM4")
        self.assertEqual(serial_factory.call_args.kwargs["baudrate"], 115200)


if __name__ == "__main__":
    unittest.main()
