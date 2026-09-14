"""관절 각도와 모터 패킷 사이의 변환 및 명령 스크립트를 실제 제조사 SDK로 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from hardware.joint_control import JointCalibration, JointController, JointState
from hardware.sts3215 import MotorError, STS3215Bus
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
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", Path(directory.name) / "logs")
        log_patch.start()
        self.addCleanup(log_patch.stop)
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

    def test_torque_read_uses_joint_mapping_and_never_writes(self) -> None:
        """관절 이름에 해당하는 모터만 조회하고 켜짐·꺼짐을 쓰기 없이 구분한다."""
        for name, servo_id, torque in (("J1", 1, 0), ("J1", 1, 1), ("J2", 2, 1)):
            with self.subTest(joint=name, torque=torque):
                self.serial.registers[5] = servo_id
                self.serial.registers[40] = torque
                self.assertIs(self.controller.read_torque(name), bool(torque))
                packet = self.serial.packets[-1]
                self.assertEqual(packet[2], servo_id)
                self.assertEqual(packet[4:7], bytes([2, 40, 1]))
        self.assertEqual(len(self.serial.packets), 3)
        with self.assertRaises(ValueError):
            self.controller.read_torque("unknown")
        self.assertEqual(len(self.serial.packets), 3)

    def test_invalid_torque_or_failed_read_is_not_reported_as_off(self) -> None:
        """알 수 없는 값과 통신 오류를 정상적인 토크 해제로 바꾸지 않는다."""
        self.serial.registers[40] = 128
        with self.assertRaisesRegex(MotorError, "토크 상태"):
            self.controller.read_torque("J1")
        self.serial.registers[40] = 0
        for failure in ("checksum", "disconnect"):
            with self.subTest(failure=failure), self.assertRaises(MotorError):
                self.serial.failure = failure
                self.controller.read_torque("J1")
        self.assertTrue(all(packet[4] == 2 for packet in self.serial.packets))

    def test_invalid_command_never_writes_motor(self) -> None:
        """단위나 관절이 잘못된 명령은 토크를 켜거나 목표를 쓰지 않는다."""
        for joint, angle, speed in [("unknown", 5, 10), ("J1", 100, 10), ("J1", 5, 100)]:
            with self.subTest(joint=joint, angle=angle), self.assertRaises(ValueError):
                self.controller.move_to(joint, angle, speed_deg_s=speed)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_arrival_uses_shared_angle_tolerance_and_requires_rest(self) -> None:
        """허용 오차 경계와 정지 조건을 공통 모듈에서 판정하며 잘못된 상태를 거부한다."""
        self.assertEqual(self.controller.tolerance_deg, 0.263671875)
        self.assertEqual(self.controller.tolerance_for("J1"), 0.1318359375)
        self.assertEqual(self.controller.tolerance_for("J2"), 0.263671875)
        for position, speed, expected in [(0.1318359375, 0.0, True), (-0.1318359375, 0.0, True),
                                          (0.13184, 0.0, False), (0.17578125, 0.0, False),
                                          (0.0, 0.01, False)]:
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
        self.assertTrue(tolerances)
        self.assertTrue(all(tolerance == 0.25 for tolerance in tolerances))

    def test_duplicate_ids_fail_before_communication(self) -> None:
        """중복된 관절 ID는 통신 전에 발견한다."""
        self.document["joints"]["J2"]["servo_id"] = 1
        self.path.write_text(yaml.safe_dump(self.document))
        with self.assertRaisesRegex(ValueError, "중복"):
            JointController(self.bus, self.path)
        self.assertEqual(self.serial.packets, [])

    def test_stopped_outside_three_counts_is_not_arrival(self) -> None:
        """정지한 네 카운트 오차를 거부하고 세 카운트 이내에서 이동을 완료한다."""
        self.controller.move_to("J1", 5)
        target = int.from_bytes(self.serial.registers[42:44], "little")
        writes = [packet for packet in self.serial.packets if packet[4] == 3]
        for offset, status in ((4, "moving"), (3, "arrived")):
            with self.subTest(offset=offset):
                self.serial.registers[56:58] = (target + offset).to_bytes(2, "little")
                state = self.controller.read("J1")
                self.assertEqual(state.motion_status, status)
                self.assertEqual(state.tolerance_deg, self.controller.tolerance_for("J1"))
        self.assertEqual([packet for packet in self.serial.packets if packet[4] == 3], writes)
        self.assertEqual(int.from_bytes(self.serial.registers[42:44], "little"), target)

    def test_completed_move_does_not_monitor_manual_displacement(self) -> None:
        """이동 완료 뒤에는 실제 각도만 갱신하고 외력에 대한 추가 판정은 하지 않는다."""
        self.controller.move_to("J1", 5)
        completed = self.controller.read("J1")
        self.assertEqual(completed.motion_status, "arrived")
        self.serial.registers[56:58] = (2048).to_bytes(2, "little")
        with patch.object(self.controller, "has_arrived") as judge:
            state = self.controller.read("J1")
            judge.assert_not_called()
        self.assertEqual(state.position_deg, 0)
        self.assertEqual(state.motion_status, "arrived")
        self.assertEqual(state.error_deg, completed.error_deg)

    def test_timeout_keeps_original_target_without_extra_commands(self) -> None:
        """기한을 넘기면 미도달로 처리하고 목표를 늘리거나 덮어쓰지 않는다."""
        with patch("hardware.joint_control.time.monotonic", return_value=100.0) as clock:
            self.controller.move_to("J1", 5)
            target = int.from_bytes(self.serial.registers[42:44], "little")
            self.serial.registers[56:58] = (target + 7).to_bytes(2, "little")
            clock.return_value = 110.0
            state = self.controller.read("J1")
            self.assertEqual(state.motion_status, "timeout")
            self.assertEqual(int.from_bytes(self.serial.registers[42:44], "little"), target)

    def test_slow_move_gets_time_for_commanded_distance(self) -> None:
        """느린 장거리 이동을 고정된 짧은 기한으로 실패 처리하지 않는다."""
        with patch("hardware.joint_control.time.monotonic", return_value=100.0) as clock:
            self.controller.move_to("J1", 20, speed_deg_s=1)
            self.serial.registers[56:58] = (2048).to_bytes(2, "little")
            clock.return_value = 106.0
            self.assertEqual(self.controller.read("J1").motion_status, "moving")
            clock.return_value = 123.0
            self.assertEqual(self.controller.read("J1").motion_status, "timeout")

    def test_stop_and_torque_changes_cancel_tracking(self) -> None:
        """명시적인 정지와 토크 변경 후 옛 이동 판정을 되살리지 않는다."""
        self.controller.move_to("J1", 5)
        self.controller.stop("J1")
        self.assertIsNone(self.controller.read("J1").target_deg)
        self.controller.set_torque("J1", False)
        self.serial.registers[56:58] = (2000).to_bytes(2, "little")
        self.controller.set_torque("J1", True)
        state = self.controller.read("J1")
        self.assertIsNone(state.target_deg)
        self.assertEqual(state.motion_status, "idle")
        self.controller.set_torque("J1", False)
        self.assertEqual(self.controller.read("J1").motion_status, "idle")

    def test_reloaded_calibration_invalidates_old_goal(self) -> None:
        """캘리브레이션 파일을 다시 읽으면 이전 이동의 목표를 사용하지 않는다."""
        self.controller.move_to("J1", 5)
        self.document["joints"]["J1"]["home_raw"] = 1934
        self.path.write_text(yaml.safe_dump(self.document))
        self.controller.reload_calibration()
        self.assertIsNone(self.controller.read("J1").target_deg)

    def test_read_failure_and_failed_new_command_do_not_reuse_arrival(self) -> None:
        """읽기 실패를 도착으로 처리하지 않고 실패한 새 명령 뒤에 이전 성공을 재사용하지 않는다."""
        self.controller.move_to("J1", 5)
        self.assertEqual(self.controller.read("J1").motion_status, "arrived")
        self.serial.failure = "checksum"
        with self.assertRaises(MotorError):
            self.controller.read("J1")
        self.serial.failure = None
        with patch.object(self.bus, "move_to", side_effect=MotorError("전송 실패")):
            with self.assertRaises(MotorError):
                self.controller.move_to("J1", 10)
        self.assertEqual(self.controller.read("J1").motion_status, "idle")

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
