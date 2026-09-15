"""가짜 모터 아홉 개와 실제 제조사 SDK로 중점·영점 설정 및 실패 복구를 검증한다."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from hardware.calibration import MotorCalibration
from hardware.joint_control import DEFAULT_CALIBRATION_PATH, JointController
from hardware.sts3215 import MotorError, STS3215Bus
from test_sts3215 import FakeSerial


class FakeMotorChain(FakeSerial):
    """등록된 ID만 응답하며 중점 명령과 응답 누락을 재현하는 가짜 직렬 버스다."""

    def __init__(self) -> None:
        """서로 다른 현재 위치를 가진 모터 아홉 개를 준비한다."""
        super().__init__()
        self.devices = {}
        for servo_id, position in enumerate((4050, 3172, 3642, 1669, 1998, 2670, 1622, 122, 1030)):
            registers = self.registers.copy()
            registers[5] = servo_id
            registers[56:58] = position.to_bytes(2, "little")
            self.devices[servo_id] = registers
        self.fail_midpoint_id = None
        self.ignore_midpoint = False

    def write(self, packet: list[int]) -> int:
        """해당 ID의 장치에 패킷을 전달하고 중점 설정은 위치 명령 없이 반영한다."""
        if packet[2] == 254 and packet[4] == 0x83:
            self.packets.append(bytes(packet))
            if sum(packet[2:]) & 0xFF != 0xFF:
                raise AssertionError("동기 쓰기 체크섬이 올바르지 않습니다.")
            if self.failure == "disconnect":
                raise OSError("시험용 연결 끊김")
            address, size = packet[5:7]
            for offset in range(7, len(packet) - 1, size + 1):
                registers = self.devices.get(packet[offset])
                if registers is not None:
                    registers[address:address + size] = bytes(packet[offset + 1:offset + 1 + size])
                    if registers[40] == 1:
                        registers[56:58] = registers[42:44]
            return len(packet)
        servo_id = packet[2]
        if servo_id not in self.devices:
            self.packets.append(bytes(packet))
            return len(packet)
        self.registers = self.devices[servo_id]
        result = super().write(packet)
        if bytes(packet[4:7]) == bytes([3, 40, 128]):
            if not self.ignore_midpoint:
                self.registers[56:58] = (2048).to_bytes(2, "little")
            if servo_id == self.fail_midpoint_id:
                self.response.clear()
        return result


class CalibrationTests(unittest.TestCase):
    """실물이나 실제 영점 파일을 건드리지 않고 설정 절차를 확인한다."""

    def setUp(self) -> None:
        """임시 영점 파일과 가짜 직렬 버스에 두 모듈을 연결한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        log_directory = tempfile.TemporaryDirectory()
        self.addCleanup(log_directory.cleanup)
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", Path(log_directory.name))
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.path = Path(directory.name) / "calibration.yaml"
        self.document = yaml.safe_load(DEFAULT_CALIBRATION_PATH.read_text())
        self.path.write_text(yaml.safe_dump(self.document, sort_keys=False))
        self.serial = FakeMotorChain()
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)
        self.calibration = MotorCalibration(self.bus, self.path)
        self.controller = JointController(self.bus, self.path)

    def _writes(self) -> list[bytes]:
        """모터에 실제 전송한 쓰기 명령만 반환한다."""
        return [packet for packet in self.serial.packets if packet[4] == 3]

    def test_all_nine_midpoints_and_zeros_persist_without_motion(self) -> None:
        """아홉 모터의 중점과 파일 영점을 확인하고 토크·다른 설정은 유지한다."""
        names = self.calibration.calibrate_all_zero()
        self.assertEqual(names, self.controller.joint_names)
        expected = deepcopy(self.document)
        for name in names:
            expected["joints"][name]["home_raw"] = 2048
        self.assertEqual(yaml.safe_load(self.path.read_text()), expected)
        writes = self._writes()
        self.assertEqual([(packet[2], packet[5], packet[6]) for packet in writes],
                         [(servo_id, 40, value) for servo_id in range(9) for value in (128, 0)])
        restarted = JointController(self.bus, self.path)
        self.assertEqual([state.position_deg for state in restarted.read_all()], [0.0] * 9)
        self.assertTrue(all(not restarted.read_torque(name) for name in names))

    def test_preflight_failure_on_last_motor_never_changes_any_motor_or_file(self) -> None:
        """마지막 모터가 응답하지 않아도 앞선 모터를 먼저 변경하지 않는다."""
        before = self.path.read_bytes()
        del self.serial.devices[8]
        with self.assertRaises(MotorError):
            self.calibration.calibrate_all_zero()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self._writes(), [])

    def test_midpoint_requires_normal_mode_torque_off_and_rest(self) -> None:
        """토크·이동·멀티턴·모드 이상을 중점 쓰기 전에 거부한다."""
        for address, value in ((40, 1), (58, 100), (18, 16), (33, 1)):
            with self.subTest(address=address):
                self.serial.devices[0][address] = value
                try:
                    with self.assertRaises(ValueError):
                        self.bus.calibrate_midpoint(0)
                    self.assertEqual(self._writes(), [])
                finally:
                    self.serial.devices[0][address] = 0

    def test_readback_failure_keeps_pending_and_torque_off(self) -> None:
        """중점 명령을 무시하는 장치를 성공으로 처리하지 않는다."""
        self.serial.ignore_midpoint = True
        with patch("hardware.sts3215.time.monotonic", side_effect=[0.0, 1.0]):
            with self.assertRaisesRegex(MotorError, "확인하지 못했습니다"):
                self.calibration.calibrate_all_zero()
        self.assertTrue(yaml.safe_load(self.path.read_text())["midpoint_pending"])
        self.assertEqual(self.serial.devices[0][40], 0)

    def test_partial_failure_blocks_old_reference_after_restart_and_retry_recovers(self) -> None:
        """응답 누락 후 이전 영점 사용을 막고 명시적인 재실행으로 복구한다."""
        self.serial.fail_midpoint_id = 4
        with self.assertRaisesRegex(MotorError, "미완료"):
            self.calibration.calibrate_all_zero()
        writes = self._writes()
        self.assertEqual([packet[2] for packet in writes if packet[6] == 128], list(range(5)))
        self.assertTrue(all(registers[40] == 0 for registers in self.serial.devices.values()))
        restarted = JointController(self.bus, self.path)
        for command in (lambda: restarted.read("J1"), lambda: restarted.move_to("J1", 0),
                        lambda: restarted.set_torque("J1", True)):
            with self.assertRaisesRegex(MotorError, "완료되지"):
                command()
        restarted.set_torque("J1", False)
        with self.assertRaisesRegex(MotorError, "미완료"):
            self.calibration.save_zero("J1")
        self.serial.fail_midpoint_id = None
        self.calibration.calibrate_all_zero()
        restarted.reload_calibration()
        self.assertEqual([state.position_deg for state in restarted.read_all()], [0.0] * 9)

    def test_initial_save_failure_never_writes_motors(self) -> None:
        """미완료 표시를 저장할 수 없으면 하드웨어 설정을 시작하지 않는다."""
        before = self.path.read_bytes()
        with patch("hardware.calibration.Path.replace", side_effect=OSError("저장 실패")):
            with self.assertRaises(OSError):
                self.calibration.calibrate_all_zero()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self._writes(), [])
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_final_save_failure_preserves_pending_on_disk(self) -> None:
        """모터 설정 후 영점 파일 교체에 실패하면 재시작해도 완료로 표시하지 않는다."""
        replace = Path.replace
        calls = []

        def fail_final(source: Path, destination: Path) -> Path:
            """두 번째 파일 교체만 실패시킨다."""
            calls.append(source)
            if len(calls) == 2:
                raise OSError("최종 저장 실패")
            return replace(source, destination)

        with patch("hardware.calibration.Path.replace", fail_final):
            with self.assertRaisesRegex(MotorError, "최종 저장 실패"):
                self.calibration.calibrate_all_zero()
        self.assertTrue(yaml.safe_load(self.path.read_text())["midpoint_pending"])
        with self.assertRaises(MotorError):
            JointController(self.bus, self.path).read_all()

    def test_software_zero_only_changes_selected_reference(self) -> None:
        """소프트웨어 영점 저장은 해당 관절의 파일 기준만 바꾼다."""
        self.calibration.save_zero("J1")
        expected = deepcopy(self.document)
        expected["joints"]["J1"]["home_raw"] = 3172
        self.assertEqual(yaml.safe_load(self.path.read_text()), expected)
        self.controller.reload_calibration()
        self.assertEqual(self.controller.read("J1").position_deg, 0)
        self.assertEqual(self._writes(), [])

    def test_software_zero_failure_preserves_file_and_reference(self) -> None:
        """파일 쓰기 실패와 움직이는 모터에서 기존 영점을 보존한다."""
        before = self.path.read_bytes()
        with patch("hardware.calibration.Path.replace", side_effect=OSError("저장 실패")):
            with self.assertRaises(OSError):
                self.calibration.save_zero("J1")
        self.serial.devices[1][58] = 100
        with self.assertRaisesRegex(ValueError, "움직이고"):
            self.calibration.save_zero("J1")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self._writes(), [])

    def test_external_file_change_is_not_overwritten(self) -> None:
        """다른 실행에서 바꾼 방향 등의 설정을 이전 값으로 덮어쓰지 않는다."""
        changed = deepcopy(self.document)
        changed["joints"]["J1"]["direction"] *= -1
        self.path.write_text(yaml.safe_dump(changed))
        with self.assertRaisesRegex(ValueError, "파일이 변경"):
            self.calibration.calibrate_all_zero()
        self.assertEqual(self._writes(), [])
        self.assertEqual(yaml.safe_load(self.path.read_text()), changed)


if __name__ == "__main__":
    unittest.main()
