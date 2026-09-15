"""가짜 모터에서 위치 게인의 변경 범위·설정 잠금·실패 시 로그 갱신을 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware.joint_control import JointController
from hardware.sts3215 import MotorError, STS3215Bus
from test_calibration import FakeMotorChain


class MotorGainTests(unittest.TestCase):
    """실제 제조사 패킷을 사용하되 물리 장치 대신 가짜 모터에 전달한다."""

    def setUp(self) -> None:
        """각 모터의 기본 게인과 독립된 로그 폴더를 준비한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", Path(directory.name))
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.serial = FakeMotorChain()
        for registers in self.serial.devices.values():
            registers[21:24] = bytes((32, 32, 0))
            registers[40] = 1
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)

    def test_only_requested_gain_changes_and_fresh_settings_are_logged(self) -> None:
        """J2의 P만 쓰고 목표·토크·그리퍼를 보존하며 로그의 이전 설정 캐시를 갱신한다."""
        before = {key: bytes(value) for key, value in self.serial.devices.items()}
        controller = JointController(self.bus)
        controller.read("J2")
        result = controller.set_position_gains("J2", p=40, i=0, d=32)
        self.assertEqual(result["before"]["pid_p"], 32)
        self.assertEqual(result["after"]["pid_p"], 40)
        self.assertEqual(result["changed_registers"], [21])
        writes = [(packet[2], packet[5], packet[6]) for packet in self.serial.packets if packet[4] == 3]
        self.assertEqual(writes, [(2, 55, 0), (2, 21, 40), (2, 55, 1)])
        for servo_id, registers in self.serial.devices.items():
            expected = bytearray(before[servo_id])
            if servo_id == 2:
                expected[21] = 40
            self.assertEqual(registers, expected)
        records = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
        settings = [row for row in records if row["event"] == "settings" and row["joint"] == "J2"]
        self.assertEqual([row["pid_p"] for row in settings], [32, 40])
        self.assertTrue(any(row["event"] == "gain_change_result" and row["result"] == "applied" for row in records))
        self.assertIsNone(controller.command_id("J2"))

    def test_unchanged_gains_do_not_rewrite_memory(self) -> None:
        """동일한 게인을 다시 확인할 때 EEPROM에 불필요하게 쓰지 않는다."""
        result = self.bus.set_position_gains(2, p=32, i=0, d=32)
        self.assertEqual(result["changed_registers"], [])
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_invalid_or_moving_motor_is_not_written(self) -> None:
        """범위 밖 게인과 이동 중 변경은 쓰기 전에 거부한다."""
        with self.assertRaises(ValueError):
            self.bus.set_position_gains(2, p=True, i=0, d=32)
        with self.assertRaises(ValueError):
            self.bus.set_position_gains(2, p=255, i=0, d=32)
        self.serial.devices[2][58:60] = (100).to_bytes(2, "little")
        with self.assertRaises(ValueError):
            self.bus.set_position_gains(2, p=40, i=0, d=32)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_ignored_gain_write_is_detected_and_relocked(self) -> None:
        """쓰기 응답만 성공하고 값이 바뀌지 않은 경우 실패로 처리한 뒤 잠금을 복구한다."""
        write = self.bus.write_register

        def ignore_gain(servo_id: int, address: int, value: int) -> None:
            """P 쓰기만 무시하고 설정 잠금은 실제 가짜 패킷으로 처리한다."""
            if address != 21:
                write(servo_id, address, value)

        with patch.object(self.bus, "write_register", side_effect=ignore_gain):
            with self.assertRaises(MotorError):
                self.bus.set_position_gains(2, p=40, i=0, d=32)
        self.assertEqual(self.serial.devices[2][21], 32)
        self.assertEqual(self.serial.devices[2][55], 1)

    def test_uncertain_write_refreshes_actual_gain_in_log(self) -> None:
        """쓰기 후 응답이 유실돼도 실제 바뀐 값을 다시 기록해 이전 게인으로 오인하지 않는다."""
        write = self.bus.write_register

        def lose_reply(servo_id: int, address: int, value: int) -> None:
            """모터에는 값을 적용하고 P 쓰기 응답만 유실시킨다."""
            write(servo_id, address, value)
            if address == 21:
                raise MotorError("쓰기 응답 유실")

        controller = JointController(self.bus)
        controller.read("J2")
        with patch.object(self.bus, "write_register", side_effect=lose_reply):
            with self.assertRaises(MotorError):
                controller.set_position_gains("J2", p=40, i=0, d=32)
        records = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
        self.assertEqual(records[-1]["event"], "settings")
        self.assertEqual(records[-1]["pid_p"], 40)
        self.assertEqual(self.serial.devices[2][55], 1)
