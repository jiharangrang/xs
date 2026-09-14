"""관절 각도와 모터 패킷 사이의 변환 및 영점 저장을 실제 제조사 SDK로 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from hardware.joint_control import JointCalibration, JointController, JointState
from hardware.sts3215 import STS3215Bus
from scripts.test_motor import main
from test_sts3215 import FakeSerial


class JointConversionTests(unittest.TestCase):
    """영점·방향·추가 감속비와 양자화·범위 처리를 확인한다."""

    def setUp(self) -> None:
        """방향이 반대이고 추가 감속비가 있는 관절을 준비한다."""
        self.calibration = JointCalibration(1, -1, 2.0, 2048, -80.0, 80.0, 30.0)

    def test_known_position_speed_and_acceleration(self) -> None:
        """알려진 각도와 속도가 같은 관절 방향과 단위로 변환되는지 확인한다."""
        calibration = self.calibration
        self.assertEqual(calibration.degrees_to_raw(45), 1024)
        self.assertEqual(calibration.raw_to_degrees(1024), 45)
        self.assertEqual(calibration.raw_speed_to_degrees(-512), 22.5)
        self.assertEqual(calibration.speed_to_raw(22.5), 512)
        self.assertEqual(calibration.acceleration_to_raw(562.5), 128)

    def test_round_trip_error_is_at_most_half_a_count(self) -> None:
        """방향과 감속비가 달라도 왕복 변환 오차가 반 카운트 이내인지 확인한다."""
        for direction in (-1, 1):
            for gear_ratio in (0.5, 1.0, 2.0):
                calibration = replace(self.calibration, direction=direction, gear_ratio=gear_ratio)
                for angle in (-10.123, 0.0, 19.876):
                    actual = calibration.raw_to_degrees(calibration.degrees_to_raw(angle))
                    self.assertLessEqual(abs(actual - angle), 0.5 / calibration.counts_per_degree + 1e-12)

    def test_single_turn_boundary_never_wraps_or_clamps(self) -> None:
        """영점이 경계에 가까우면 반대쪽으로 접지 않고 실행 불가능한 목표를 거부한다."""
        calibration = replace(self.calibration, direction=1, gear_ratio=1.0, zero_raw=4090)
        with self.assertRaisesRegex(ValueError, "단회전 경계"):
            calibration.degrees_to_raw(5)
        self.assertEqual(calibration.degrees_to_raw(0), 4090)
        self.assertLess(calibration.raw_to_degrees(10), -350)

    def test_invalid_angles_rates_and_calibration_are_rejected(self) -> None:
        """숫자 이상·관절 범위 초과·표현할 수 없는 속도와 잘못된 설정을 거부한다."""
        for value in (float("nan"), float("inf"), True, 81.0):
            with self.subTest(angle=value), self.assertRaises(ValueError):
                self.calibration.degrees_to_raw(value)
        for value in (-1, 0, 0.0001, 31, float("nan")):
            with self.subTest(speed=value), self.assertRaises(ValueError):
                self.calibration.speed_to_raw(value)
        for value in (0, -1, 0.0001, 1e6, float("inf")):
            with self.subTest(acceleration=value), self.assertRaises(ValueError):
                self.calibration.acceleration_to_raw(value)
        for change in ({"direction": 0}, {"direction": True}, {"gear_ratio": 0},
                       {"zero_raw": 4096}, {"servo_id": True}, {"max_speed_deg_s": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.calibration, **change)


class JointControllerTests(unittest.TestCase):
    """도 단위 통로가 실제 패킷과 영점 파일까지 올바르게 연결되는지 확인한다."""

    def setUp(self) -> None:
        """실물과 분리한 직렬 포트와 임시 관절 설정 파일을 준비한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "calibration.yaml"
        self.document = {
            "model": "test",
            "joint_order": ["J1", "J2"],
            "joints": {
                "J1": {"servo_id": 1, "direction": -1, "gear_ratio": 2.0,
                       "home_raw": None, "home_single": 2048, "lower_rad": -1.4,
                       "upper_rad": 1.4, "max_speed_rad_s": 0.55, "leader_id": 7},
                "J2": {"servo_id": 2, "direction": 1, "gear_ratio": 1.0,
                       "home_raw": None, "home_single": 2048, "lower_rad": -2.9,
                       "upper_rad": 2.9, "max_speed_rad_s": 0.55},
            },
        }
        self.path.write_text(yaml.safe_dump(self.document, sort_keys=False), encoding="utf-8")
        self.serial = FakeSerial()
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)
        self.controller = JointController(self.bus, self.path)

    def test_degree_command_reaches_sdk_with_correct_direction_and_units(self) -> None:
        """양의 관절 명령이 반대 방향 모터 목표와 도 단위 피드백으로 이어지는지 확인한다."""
        target = self.controller.move_to("J1", 5)
        self.assertEqual(self.serial.registers[42:44], (1934).to_bytes(2, "little"))
        self.assertEqual(self.serial.packets[-1][5:-1], bytes([41, 20, 142, 7, 0, 0, 228, 0]))
        self.assertEqual(self.serial.position_on_enable, 2048)
        self.assertAlmostEqual(target, 5.009765625)
        self.serial.registers[58:60] = (0x80E4).to_bytes(2, "little")
        state = self.controller.read("J1")
        self.assertEqual(state.name, "J1")
        self.assertAlmostEqual(state.position_deg, target)
        self.assertAlmostEqual(state.speed_deg_s, 10.01953125)
        self.controller.move_by("J1", 5)
        self.assertEqual(self.serial.registers[42:44], (1820).to_bytes(2, "little"))

    def test_invalid_command_never_writes_motor(self) -> None:
        """단위나 관절이 잘못된 명령은 토크를 켜거나 목표를 쓰지 않는다."""
        for joint, angle, speed in [("unknown", 5, 10), ("J1", 100, 10), ("J1", 5, 100)]:
            with self.subTest(joint=joint, angle=angle), self.assertRaises(ValueError):
                self.controller.move_to(joint, angle, speed_deg_s=speed)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_arrival_uses_shared_angle_tolerance_and_requires_rest(self) -> None:
        """허용 오차 경계와 정지 조건을 공통 모듈에서 판정하며 잘못된 상태를 거부한다."""
        for position, speed, expected in [(1.0, 0.0, True), (-1.0, 0.0, True),
                                          (1.0001, 0.0, False), (0.0, 0.01, False)]:
            with self.subTest(position=position, speed=speed):
                self.assertEqual(self.controller.has_arrived(JointState("J1", position, speed), 0), expected)
        with patch("hardware.joint_control.DEFAULT_TOLERANCE_DEG", 0.25):
            controller = JointController(self.bus, self.path)
            self.assertFalse(controller.has_arrived(JointState("J1", 0.3, 0), 0))
            self.assertEqual(controller.tolerance_deg, 0.25)
        for tolerance in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                JointController(self.bus, self.path, tolerance_deg=tolerance)
        with self.assertRaises(ValueError):
            self.controller.has_arrived(JointState("J1", float("nan"), 0), 0)
        self.assertEqual(self.serial.packets, [])

    def test_cli_delegates_arrival_and_override_to_controller(self) -> None:
        """스크립트가 자체 판정 없이 공통 제어기에 지정한 허용 오차를 전달한다."""
        tolerances = []
        original = JointController.has_arrived

        def observe(controller: JointController, state: JointState, target_deg: float) -> bool:
            """실제 공통 판정에 사용된 허용 오차를 기록한다."""
            tolerances.append(controller.tolerance_deg)
            return original(controller, state, target_deg)

        with patch.object(JointController, "has_arrived", autospec=True, side_effect=observe), redirect_stdout(StringIO()):
            self.assertEqual(main(["move", "--port", "FAKE", "--joint", "J1", "--calibration", str(self.path),
                                   "--delta-deg", "5", "--tolerance-deg", "0.25"]), 0)
        self.assertEqual(tolerances, [0.25])

    def test_save_zero_persists_only_selected_reference_without_motor_write(self) -> None:
        """현재 자세를 저장하고 다시 열어도 영점이 유지되며 나머지 설정은 보존한다."""
        self.serial.registers[56:58] = (1934).to_bytes(2, "little")
        self.controller.save_zero("J1")
        saved = yaml.safe_load(self.path.read_text())
        expected = self.document
        expected["joints"]["J1"]["home_raw"] = 1934
        self.assertEqual(saved, expected)
        self.assertEqual(JointController(self.bus, self.path).read("J1").position_deg, 0)
        self.assertEqual(self.controller.read("J1").position_deg, 0)
        self.assertTrue(all(packet[4] == 2 for packet in self.serial.packets))

    def test_zero_failure_preserves_file_and_current_reference(self) -> None:
        """영점 파일 교체 실패 시 기존 파일과 메모리의 기준을 그대로 유지한다."""
        before = self.path.read_bytes()
        self.serial.registers[56:58] = (1934).to_bytes(2, "little")
        with patch("hardware.joint_control.Path.replace", side_effect=OSError("저장 실패")):
            with self.assertRaises(OSError):
                self.controller.save_zero("J1")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertAlmostEqual(self.controller.read("J1").position_deg, 5.009765625)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_moving_motor_cannot_be_saved_as_zero(self) -> None:
        """움직이는 중에는 새 영점을 저장하지 않는다."""
        before = self.path.read_bytes()
        self.serial.registers[58:60] = (100).to_bytes(2, "little")
        with self.assertRaisesRegex(ValueError, "움직이고"):
            self.controller.save_zero("J1")
        self.assertEqual(self.path.read_bytes(), before)

    def test_duplicate_ids_fail_before_communication(self) -> None:
        """중복된 관절 ID는 통신 전에 발견한다."""
        self.document["joints"]["J2"]["servo_id"] = 1
        self.path.write_text(yaml.safe_dump(self.document))
        with self.assertRaisesRegex(ValueError, "중복"):
            JointController(self.bus, self.path)
        self.assertEqual(self.serial.packets, [])

    def test_cli_uses_degrees_and_rejects_old_raw_flags(self) -> None:
        """각도 CLI가 도 단위 결과를 출력하고 이전 raw 옵션의 자동 해석을 막는다."""
        output = StringIO()
        args = ["--port", "FAKE", "--joint", "J1", "--calibration", str(self.path)]
        with redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(["move", *args, "--delta-deg", "5"]), 0)
        self.assertIn("5.010°", output.getvalue())
        self.assertIn("°/s", output.getvalue())
        self.assertNotIn("raw", output.getvalue())
        for old_flag in ("--delta", "--position", "--speed", "--acceleration"):
            with self.subTest(old_flag=old_flag), redirect_stderr(StringIO()):
                self.serial.packets.clear()
                with self.assertRaises(SystemExit) as error:
                    main(["move", *args, old_flag, "50"])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(self.serial.packets, [])

    def test_cli_zero_and_id_zero_use_the_same_channel(self) -> None:
        """ID 0 선택과 영점 저장도 같은 관절 통로를 사용한다."""
        self.document["joints"]["J1"]["servo_id"] = 0
        self.path.write_text(yaml.safe_dump(self.document))
        self.serial.registers[5] = 0
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["zero", "--port", "FAKE", "--id", "0", "--calibration", str(self.path)]), 0)
        self.assertIn("0°로 저장", output.getvalue())
        self.assertEqual(yaml.safe_load(self.path.read_text())["joints"]["J1"]["home_raw"], 2048)
        self.assertTrue(all(packet[4] == 2 for packet in self.serial.packets))


if __name__ == "__main__":
    unittest.main()
