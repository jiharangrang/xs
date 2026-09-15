"""자세 묶음 명령의 실제 SDK 패킷·도착 오차·로그와 터미널 입력을 실물 없이 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hardware.joint_control import JointController
from hardware.sts3215 import MotorError, STS3215Bus
from kinematics.joints import ARM_JOINT_NAMES
from scripts.move_pose import PoseInterrupted, main, observe_pose, parse_angles, print_result
from test_calibration import FakeMotorChain


class PoseMotionTests(unittest.TestCase):
    """변환·일괄 검증·전송·관측이 하나의 제어 경로로 이어지는지 확인한다."""

    def setUp(self) -> None:
        """실제 SDK 아래에 가짜 모터 아홉 개와 임시 로그를 연결한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        logger = patch("hardware.motor_logging.DEFAULT_LOG_DIR", Path(directory.name))
        logger.start()
        self.addCleanup(logger.stop)
        self.serial = FakeMotorChain()
        for registers in self.serial.devices.values():
            registers[56:58] = (2048).to_bytes(2, "little")
        serial = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial.start()
        self.addCleanup(serial.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)
        self.controller = JointController(self.bus)

    def test_one_packet_moves_seven_joints_and_preserves_grippers_and_logs(self) -> None:
        """일곱 목표가 같은 패킷에 들어가며 변환·속도·로그와 그리퍼 유지가 맞는다."""
        before = {joint: bytes(self.serial.devices[joint]) for joint in (0, 8)}
        angles = dict(zip(ARM_JOINT_NAMES, (5, -5, 5, 10, 5, 10, 5)))
        targets = self.controller.move_many(angles)
        packets = [packet for packet in self.serial.packets if packet[4] == 0x83]
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0][2], 254)
        self.assertEqual(packets[0][5:7], bytes([41, 7]))
        self.assertEqual(list(packets[0][7:-1:8]), list(range(1, 8)))
        for servo_id, name in enumerate(ARM_JOINT_NAMES, 1):
            registers = self.serial.devices[servo_id]
            self.assertEqual(int.from_bytes(registers[46:48], "little"), 114)
            self.assertEqual(registers[41], 10)
            state = self.controller.read(name)
            self.assertEqual(state.motion_status, "arrived")
            self.assertEqual(state.position_deg, targets[name])
            self.assertEqual(state.error_raw, 0)
            self.assertEqual(state.tolerance_raw, 3)
            self.assertIsNotNone(state.command_id)
        self.assertEqual(before, {joint: bytes(self.serial.devices[joint]) for joint in (0, 8)})
        rows = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
        commands = [row for row in rows if row["event"] == "command"]
        self.assertEqual({row["joint"] for row in commands}, set(ARM_JOINT_NAMES))
        self.assertTrue(all(row["action"] == "move_many" for row in commands))
        for row in (row for row in rows if row["event"] == "sample"):
            self.assertIn("current_ma", row)
            self.assertIn("pwm_percent", row)

    def test_last_invalid_target_or_mode_never_writes_any_motor(self) -> None:
        """마지막 목표나 장치 모드가 잘못되어도 앞 관절을 먼저 움직이지 않는다."""
        cases = ({"J1": 5, "J2": 26}, {"J1": 5, "J4": -31},
                 {"J1": 5, "J6": -16}, {"J1": 5, "typo": 0}, {"J1": True})
        for targets in cases:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.controller.move_many(targets)
        self.serial.devices[7][33] = 1
        with self.assertRaises(ValueError):
            self.controller.move_many(dict.fromkeys(ARM_JOINT_NAMES, 5))
        self.assertFalse(any(packet[4] in (3, 0x83) for packet in self.serial.packets))

    def test_raw_arrival_boundary_and_timeout_do_not_send_correction(self) -> None:
        """세 카운트는 도착, 네 카운트는 미도착으로 판정하며 보정 명령은 추가하지 않는다."""
        self.controller.move_many({"J1": 5, "J2": -5})
        for servo_id, error in ((1, 3), (2, 4)):
            registers = self.serial.devices[servo_id]
            target = int.from_bytes(registers[42:44], "little")
            registers[56:58] = (target + error).to_bytes(2, "little")
        count = len([packet for packet in self.serial.packets if packet[4] in (3, 0x83)])
        with patch("hardware.joint_control.time.monotonic", return_value=1e20):
            states = [self.controller.read(name) for name in ("J1", "J2")]
        self.assertEqual([state.motion_status for state in states], ["arrived", "timeout"])
        self.assertEqual([state.error_raw for state in states], [-3, -4])
        self.assertEqual(len([packet for packet in self.serial.packets if packet[4] in (3, 0x83)]), count)

    def test_failed_broadcast_clears_buffer_and_records_failure(self) -> None:
        """실패한 목표를 다음 묶음에 섞지 않으며 전송 실패를 관절별 로그에 남긴다."""
        with patch.object(self.bus._sdk.groupSyncWrite, "txPacket", return_value=-1001):
            with self.assertRaises(MotorError):
                self.controller.move_many({"J1": 5, "J2": 5})
        self.assertEqual(self.bus._sdk.groupSyncWrite.data_dict, {})
        self.controller.move_many({"J3": 5})
        packet = [packet for packet in self.serial.packets if packet[4] == 0x83][-1]
        self.assertEqual(list(packet[7:-1:8]), [3])
        rows = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
        failures = [row for row in rows if row.get("result") == "failed"]
        self.assertEqual({row["joint"] for row in failures}, {"J1", "J2"})


class PoseScriptTests(unittest.TestCase):
    """GUI 출력의 입력 처리와 이번 명령에 한정한 결과 표시를 확인한다."""

    def test_gui_json_preserves_grippers_by_default_and_rejects_bad_input(self) -> None:
        """몸통 전체를 받되 그리퍼는 명시한 옵션에서만 포함하고 잘못된 입력은 거부한다."""
        angles = {**dict.fromkeys(ARM_JOINT_NAMES, 5), "G_L": 0, "G_R": 0}
        self.assertEqual(set(parse_angles(json.dumps(angles))), set(ARM_JOINT_NAMES))
        self.assertEqual(parse_angles(json.dumps(angles), include_grippers=True), angles)
        for invalid in ({"J1": 0}, {**angles, "J2": True}, {**angles, "J3": float("nan")},
                        {**angles, "J9": 0}, list(angles.values())):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_angles(json.dumps(invalid))

    def test_stream_ignores_old_results_and_detects_replacement(self) -> None:
        """이전 완료 메시지는 건너뛰고 정지 등 새 명령을 이번 도착으로 오인하지 않는다."""
        old = {"name": "J1", "command_id": 1, "motion_status": "arrived"}
        current = {**old, "command_id": 2, "error_raw": 4, "tolerance_raw": 3, "motion_status": "timeout"}
        socket = Mock()
        socket.recv.side_effect = [json.dumps({"joints": [old]}), json.dumps({"joints": [current]})]
        with patch("scripts.move_pose.connect") as connect:
            connect.return_value.__enter__.return_value = socket
            self.assertEqual(observe_pose("http://localhost:18765", {"J1": 2}), [current])
            socket.recv.side_effect = [json.dumps({"joints": [{**old, "command_id": 3}]})]
            with self.assertRaisesRegex(ValueError, "중단"):
                observe_pose("http://localhost:18765", {"J1": 2})

    def test_cli_sends_once_and_reports_raw_errors_briefly(self) -> None:
        """속도 입력 없이 한 번 전송하고 결과만 짧게 표시한다."""
        angles = dict.fromkeys(ARM_JOINT_NAMES, 5)
        states = [{"name": name, "error_raw": 2, "tolerance_raw": 3, "motion_status": "arrived"}
                  for name in ARM_JOINT_NAMES]
        output = StringIO()
        with patch("scripts.move_pose.post", return_value={"command_ids": {"J1": 1}, "log_path": "test.jsonl"}) as send:
            with patch("scripts.move_pose.observe_pose", return_value=states), redirect_stdout(output):
                self.assertEqual(main([json.dumps(angles)]), 0)
        send.assert_called_once_with("http://127.0.0.1:18765", "/api/pose", {"angles_deg": angles})
        self.assertIn("도착 7/7", output.getvalue())
        self.assertIn("J1=2", output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 4)
        states[-1].update(error_raw=-6, motion_status="timeout")
        with redirect_stdout(output):
            self.assertFalse(print_result(states))
        self.assertIn("허용 초과: J7", output.getvalue())

    def test_replaced_move_does_not_stop_new_command(self) -> None:
        """GUI에서 새 명령을 내린 경우 기존 관찰기가 새 이동을 정지시키지 않는다."""
        angles = dict.fromkeys(ARM_JOINT_NAMES, 5)
        with patch("scripts.move_pose.post", return_value={"command_ids": {"J1": 1}}):
            with patch("scripts.move_pose.observe_pose", side_effect=PoseInterrupted("다른 명령")):
                with patch("scripts.move_pose.stop_targets") as stop, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(main([json.dumps(angles)]), 1)
                    stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
