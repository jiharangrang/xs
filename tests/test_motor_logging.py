"""실제 제조사 SDK와 가짜 직렬 버스로 로깅 값·명령 불변성·실패 기록을 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware.joint_control import DEFAULT_CALIBRATION_PATH, JointController
from hardware.motor_logging import MotorLogger
from hardware.sts3215 import MotorError, STS3215Bus
from scripts.test_motor import main
from test_calibration import FakeMotorChain


class MotorLoggingTests(unittest.TestCase):
    """실물이나 저장된 영점을 바꾸지 않고 실행 로그 전체 흐름을 확인한다."""

    def setUp(self) -> None:
        """임시 로그 폴더와 등록된 모터 아홉 개를 준비한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.serial = FakeMotorChain()
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", self.directory)
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)
        with redirect_stderr(StringIO()):
            self.controller = JointController(self.bus)

    def _rows(self, path: Path | None = None, event: str | None = None) -> list[dict]:
        """현재 로그를 다시 열어 완전한 JSON 행만 수집한다."""
        rows = [json.loads(line) for line in (path or self.controller.log_path).read_text().splitlines()]
        return rows if event is None else [row for row in rows if row["event"] == event]

    def _word(self, servo_id: int, address: int, value: int) -> None:
        """시험용 장치의 두 바이트 값을 변경한다."""
        self.serial.devices[servo_id][address:address + 2] = value.to_bytes(2, "little")

    def test_feedback_units_signs_and_limit_use_one_read(self) -> None:
        """방향 부호·전류·PWM 단위를 확인하고 여러 상태가 한 요청으로 읽히는지 검증한다."""
        registers = self.serial.devices[1]
        registers[40] = 1
        registers[62], registers[63], registers[66] = 120, 43, 1
        self._word(1, 42, 2162)
        self._word(1, 56, 2157)
        self._word(1, 58, 0x8000 | 100)
        self._word(1, 60, 0x400 | 800)
        self._word(1, 48, 800)
        self._word(1, 69, 100)
        before = len(self.serial.packets)
        feedback = self.bus.read_feedback(1)
        self.assertEqual(len(self.serial.packets) - before, 1)
        self.assertEqual(self.serial.packets[-1][4:7], bytes([2, 40, 31]))
        self.assertEqual(feedback["speed_raw"], -100)
        self.assertEqual(feedback["load_raw"], -800)
        self.assertEqual(feedback["pwm_percent"], -80.0)
        self.assertEqual(feedback["current_ma"], 650.0)
        self.assertEqual(feedback["voltage_v"], 12.0)
        self.assertTrue(feedback["output_at_limit"])
        self._word(1, 48, 1000)
        self.assertFalse(self.bus.read_feedback(1)["output_at_limit"])

    def test_default_controller_records_settings_and_degree_raw_error(self) -> None:
        """추가 연결 코드 없이 기본 제어기에서 설정과 도·raw 피드백을 함께 저장한다."""
        registers = self.serial.devices[1]
        registers[21:24] = bytes([32, 32, 0])
        registers[26:28] = bytes([1, 1])
        self._word(1, 28, 500)
        self._word(1, 42, 2055)
        self._word(1, 56, 2051)
        state = self.controller.read("J1")
        sample = self._rows(event="sample")[-1]
        self.assertEqual(sample["motor_error_raw"], 4)
        self.assertEqual(sample["position_deg"], state.position_deg)
        self.assertAlmostEqual(abs(sample["motor_error_deg"]), 4 * 360 / 4096)
        self.assertGreaterEqual(sample["read_received_s"], sample["read_started_s"])
        self.assertIsNone(sample["sample_interval_s"])
        settings = self._rows(event="settings")[-1]
        self.assertEqual((settings["pid_p"], settings["pid_i"], settings["pid_d"]), (32, 0, 32))
        self.assertEqual(settings["current_limit_raw"], 500)
        self.assertIn("zero_raw", settings["calibration"])
        self.assertEqual(self._rows()[0]["port"], "FAKE")
        self.assertEqual(self._rows()[0]["schema_version"], 1)

    def test_settings_are_read_once_and_refreshed_after_calibration_reload(self) -> None:
        """같은 설정을 매번 읽지 않으며 기준 재조회 시 설정 기록도 새로 남긴다."""
        self.controller.read("J1")
        self.controller.read("J1")
        self.assertEqual(len(self._rows(event="settings")), 1)
        self.assertIsNotNone(self._rows(event="sample")[-1]["sample_interval_s"])
        self.serial.devices[1][23] = 2
        self.controller.reload_calibration()
        self.controller.read("J1")
        self.assertEqual(len(self._rows(event="settings")), 2)
        self.assertEqual(self._rows(event="settings")[-1]["pid_i"], 2)
        self.assertEqual(self._rows(event="calibration")[-1]["joints"]["J1"]["zero_raw"], 2048)

    def test_logging_never_adds_motor_writes(self) -> None:
        """로깅 켜짐 여부와 무관하게 이동·정지·토크 명령 패킷이 동일하다."""
        packets = []
        for enabled in (False, True):
            serial = FakeMotorChain()
            with patch("scservo_sdk.port_handler.serial.Serial", return_value=serial):
                with STS3215Bus("FAKE") as bus, redirect_stderr(StringIO()):
                    controller = JointController(bus, logger=MotorLogger(self.directory, enabled=enabled))
                    controller.move_to("J1", 5)
                    controller.read("J1")
                    controller.move_by("J1", -1)
                    controller.stop("J1")
                    controller.set_torque("J1", False)
                    packets.append([packet for packet in serial.packets if packet[4] == 3])
        self.assertEqual(packets[0], packets[1])

    def test_completed_move_keeps_logging_current_error_without_rejudging(self) -> None:
        """도착 판정은 끝내되 이후 관측한 실제 오차는 그대로 기록한다."""
        self.controller.move_to("J1", 5)
        arrived = self.controller.read("J1")
        self.assertEqual(arrived.motion_status, "arrived")
        goal = int.from_bytes(self.serial.devices[1][42:44], "little")
        self._word(1, 56, goal + 7)
        writes = [packet for packet in self.serial.packets if packet[4] == 3]
        with patch.object(self.controller, "has_arrived") as judge:
            state = self.controller.read("J1")
            judge.assert_not_called()
        sample = self._rows(event="sample")[-1]
        self.assertEqual(sample["motor_error_raw"], -7)
        self.assertEqual(sample["arrival_error_deg"], arrived.error_deg)
        self.assertEqual(state.motion_status, "arrived")
        self.assertEqual([packet for packet in self.serial.packets if packet[4] == 3], writes)

    def test_device_fault_retains_telemetry_and_still_raises(self) -> None:
        """장치가 오류를 보고해도 함께 온 진단값을 보존하고 정상 상태로 처리하지 않는다."""
        self.controller.read("J1")
        self.serial.devices[1][65] = 0x20
        self.serial.devices[1][63] = 75
        self.serial.failure = "overheat"
        with self.assertRaises(MotorError) as caught:
            self.controller.read("J1")
        self.assertEqual(caught.exception.device_error, 0x24)
        row = self._rows(event="sample_error")[-1]
        self.assertEqual(row["feedback"]["temperature_c"], 75)
        self.assertEqual(row["feedback"]["status_raw"], 0x20)
        self.assertEqual(row["feedback"]["packet_error"], 4)

    def test_communication_failure_never_reuses_previous_sample(self) -> None:
        """통신 누락은 이전 정상값을 복사하지 않고 원인과 시각을 기록한다."""
        self.controller.read("J1")
        self.serial.failure = "checksum"
        with self.assertRaises(MotorError):
            self.controller.read("J1")
        row = self._rows(event="sample_error")[-1]
        self.assertIsNone(row["feedback"])
        self.assertIsNotNone(row["communication_result"])
        self.assertEqual(len(self._rows(event="sample")), 1)

    def test_truncated_response_is_not_decoded(self) -> None:
        """체크섬을 통과했더라도 길이가 부족한 진단 응답을 거부한다."""
        with patch.object(self.bus, "_call", return_value=[list(range(30)), 0]):
            with self.assertRaisesRegex(MotorError, "길이"):
                self.bus.read_feedback(1)

    def test_unsupported_current_is_null_with_original_register_retained(self) -> None:
        """전류 피드백 미지원 설정에서는 측정값을 영으로 가장하지 않는다."""
        self.serial.devices[1][18] = 0x20
        self._word(1, 69, 123)
        self.controller.read("J1")
        sample = self._rows(event="sample")[-1]
        self.assertFalse(sample["current_supported"])
        self.assertIsNone(sample["current_ma"])
        self.assertEqual(sample["current_raw"], 123)

    def test_failed_command_has_result_and_is_not_reported_sent(self) -> None:
        """전송 실패와 요청 인자를 같은 명령 번호로 연결한다."""
        with patch.object(self.bus, "move_to", side_effect=MotorError("송신 실패")):
            with self.assertRaises(MotorError):
                self.controller.move_to("J1", 5)
        command = self._rows(event="command")[-1]
        result = self._rows(event="command_result")[-1]
        self.assertEqual(command["command_id"], result["command_id"])
        self.assertEqual(command["angle_deg"], 5)
        self.assertEqual(result["result"], "failed")
        self.assertEqual(result["error"], "송신 실패")

    def test_file_failure_warns_once_and_move_still_succeeds(self) -> None:
        """기록 중 저장 공간 오류가 나도 기존 목표 명령을 그대로 수행한다."""
        self.controller.read("J1")
        output = StringIO()
        with patch("hardware.motor_logging.Path.open", side_effect=OSError("저장 공간 없음")), redirect_stderr(output):
            self.controller.move_to("J1", 5)
            self.controller.move_to("J1", 6)
        self.assertEqual(output.getvalue().count("로그 저장 실패"), 1)
        self.assertEqual(self.serial.devices[1][42:44], self.serial.devices[1][56:58])

    def test_cli_movement_uses_default_log_without_extra_flag(self) -> None:
        """기존 CLI 명령이 별도 로깅 옵션 없이 요청과 도착 피드백을 저장한다."""
        before = set(self.directory.glob("*.jsonl"))
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = main(["move", "--port", "FAKE", "--id", "1", "--angle-deg", "5"])
        self.assertEqual(result, 0)
        created = set(self.directory.glob("*.jsonl")) - before
        self.assertEqual(len(created), 1)
        rows = self._rows(created.pop())
        self.assertTrue(any(row["event"] == "command_result" and row["result"] == "sent" for row in rows))
        self.assertTrue(any(row["event"] == "sample" and row["motion_status"] == "arrived" for row in rows))


if __name__ == "__main__":
    unittest.main()
