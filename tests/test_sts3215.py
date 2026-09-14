"""가짜 직렬 포트에 실제 제조사 SDK를 연결해 명령 패킷과 실패 처리를 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scservo_sdk import COMM_RX_CORRUPT, COMM_RX_TIMEOUT

from hardware.motor_ids import change_motor_id, find_motor_id, main as motor_ids_main
from hardware.motor_setup import configure_single_turn, main as motor_setup_main
from hardware.ports import PortSettings
from hardware.sts3215 import MotorError, STS3215Bus
from scripts.test_motor import main


class FakeSerial:
    """읽기와 쓰기 패킷에 응답하고 토크 상태에 따라 목표 위치를 반영한다."""

    def __init__(self) -> None:
        """실제 장치와 분리된 레지스터와 응답 버퍼를 준비한다."""
        self.registers = bytearray(128)
        self.registers[3:5] = (1234).to_bytes(2, "little")
        self.registers[5] = 1
        self.registers[9:13] = bytes.fromhex("00 00 ff 0f")
        self.registers[42:44] = (3100).to_bytes(2, "little")
        self.registers[56:58] = (2048).to_bytes(2, "little")
        self.registers[55] = 1
        self.response = bytearray()
        self.packets: list[bytes] = []
        self.position_on_enable: int | None = None
        self.failure: str | None = None
        self.closed = False
        self.id_reply = "old"
        self.ignore_id_write = False
        self.ignore_lock_write = False

    def reset_input_buffer(self) -> None:
        """직렬 포트를 열 때 이전 응답을 비운다."""
        self.response.clear()

    def flush(self) -> None:
        """가짜 포트에는 전송 대기가 없으므로 바로 반환한다."""

    def close(self) -> None:
        """가짜 포트가 닫혔음을 기록한다."""
        self.closed = True

    def read(self, size: int) -> bytes:
        """요청한 길이만큼 응답을 꺼내 반환한다."""
        data = bytes(self.response[:size])
        del self.response[:size]
        return data

    def write(self, packet: list[int]) -> int:
        """명령 패킷을 해석하고 레지스터 응답 또는 주입된 오류를 돌려준다."""
        packet = bytes(packet)
        self.packets.append(packet)
        if self.failure == "disconnect":
            raise OSError("시험용 연결 끊김")
        if packet[:2] != b"\xff\xff" or sum(packet[2:]) & 0xFF != 0xFF:
            raise AssertionError("잘못된 송신 패킷입니다.")
        if packet[2] != self.registers[5]:
            return len(packet)
        payload = b""
        changes_id = False
        if packet[4] == 2:
            address, size = packet[5:7]
            payload = self.registers[address:address + size]
        elif packet[4] == 3:
            address = packet[5]
            values = packet[6:-1]
            changes_id = address == 5
            ignore_write = (
                changes_id and (self.ignore_id_write or self.registers[55] != 0)
                or address == 55 and values == b"\x01" and self.ignore_lock_write
            )
            if not ignore_write:
                self.registers[address:address + len(values)] = values
            if self.registers[40] == 1 and address in (40, 41, 42):
                self.registers[56:58] = self.registers[42:44]
                if address == 40:
                    self.position_on_enable = int.from_bytes(self.registers[56:58], "little")
        if self.failure == "timeout" or changes_id and self.id_reply == "none":
            return len(packet)
        error = 4 if self.failure == "overheat" else 0
        response_id = self.registers[5] if changes_id and self.id_reply == "new" else packet[2]
        response = bytearray([255, 255, response_id, len(payload) + 2, error])
        response.extend(payload)
        response.append(~sum(response[2:]) & 0xFF)
        if self.failure == "checksum":
            response[-1] ^= 1
        self.response.extend(response)
        return len(packet)


class STS3215Tests(unittest.TestCase):
    """통신 성공뿐 아니라 잘못된 응답과 의도하지 않은 이동을 검사한다."""

    def setUp(self) -> None:
        """포트 생성만 대체하고 제조사 패킷 생성과 해석은 그대로 실행한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", Path(directory.name) / "logs")
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.serial = FakeSerial()
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        self.serial_factory = serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE", 115200).open()
        self.addCleanup(self.bus.close)

    def test_read_only_and_signed_feedback(self) -> None:
        """읽기 명령은 토크를 바꾸지 않고 음수 속도도 올바르게 해석한다."""
        self.serial.registers[58:60] = bytes.fromhex("42 80")
        self.assertEqual(self.bus.ping(1), 1234)
        self.assertEqual(self.bus.read_position_speed(1), (2048, -66))
        self.assertTrue(all(packet[4] in (1, 2) for packet in self.serial.packets))
        self.assertEqual(self.serial_factory.call_args.kwargs["baudrate"], 115200)

    def test_move_holds_current_before_enabling_and_sends_expected_packet(self) -> None:
        """토크를 켤 때 오래된 목표로 이동하지 않고 새 위치·속도·가속도를 정확히 전송한다."""
        self.bus.move_to(1, 2098)
        self.assertEqual(self.serial.position_on_enable, 2048)
        self.assertEqual(self.serial.packets[-1], bytes.fromhex("ff ff 01 0a 03 29 0a 32 08 00 00 64 00 20"))
        self.assertEqual(self.bus.read_position(1), 2098)
        self.assertEqual(self.serial.registers[40], 1)

    def test_move_with_torque_on_does_not_insert_hold_command(self) -> None:
        """이미 토크가 켜져 있으면 중간 정지 없이 새 목표만 쓴다."""
        self.serial.registers[40] = 1
        self.bus.move_to(1, 2098)
        writes = [packet for packet in self.serial.packets if packet[4] == 3]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][5], 41)

    def test_invalid_inputs_never_write(self) -> None:
        """범위를 벗어난 위치·속도·ID는 모터 쓰기 전에 거부한다."""
        for servo_id, position, speed in [(254, 2000, 100), (1, 4096, 100), (1, -1, 100), (1, 2000, 0)]:
            with self.subTest(servo_id=servo_id, position=position, speed=speed):
                with self.assertRaises(ValueError):
                    self.bus.move_to(servo_id, position, speed=speed)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_wrong_mode_or_device_limit_never_writes(self) -> None:
        """다른 제어 모드와 장치 위치 범위를 감지해 모터 설정이나 토크를 바꾸지 않는다."""
        for mode, upper in [(1, 4095), (0, 0), (0, 2000)]:
            with self.subTest(mode=mode, upper=upper):
                self.serial.registers[33] = mode
                self.serial.registers[11:13] = upper.to_bytes(2, "little")
                with self.assertRaises(ValueError):
                    self.bus.move_to(1, 2098)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_stop_and_close_preserve_torque(self) -> None:
        """정지는 현재 위치 유지 명령이며 포트 종료가 토크 해제로 이어지지 않는다."""
        self.serial.registers[40] = 1
        self.assertEqual(self.bus.stop(1), 2048)
        self.bus.close()
        self.assertTrue(self.serial.closed)
        self.assertEqual(self.serial.registers[40], 1)
        self.assertEqual(int.from_bytes(self.serial.registers[42:44], "little"), 2048)

    def test_torque_off_works_in_other_modes(self) -> None:
        """위치 제어를 지원하지 않는 모드에서도 명시적 토크 해제는 가능하다."""
        self.serial.registers[33] = 1
        self.serial.registers[40] = 1
        self.bus.set_torque(1, False)
        self.assertEqual(self.serial.registers[40], 0)

    def test_bad_feedback_is_not_reported_as_zero_position(self) -> None:
        """응답 누락·체크섬·장치 오류·연결 끊김을 정상 위치값으로 취급하지 않는다."""
        for failure in ("timeout", "checksum", "overheat", "disconnect"):
            with self.subTest(failure=failure):
                self.serial.failure = failure
                with self.assertRaises(MotorError):
                    self.bus.read_position(1)
                self.assertFalse(self.bus._port.is_using)

    def test_cli_default_read_and_relative_move(self) -> None:
        """기본 실행은 읽기만 수행하고 상대 이동은 현재 위치를 기준으로 실행한다."""
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(["--port", "FAKE", "--id", "1"]), 0)
            self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))
            self.assertEqual(main(["move", "--port", "FAKE", "--id", "1", "--delta-deg", "5"]), 0)
        self.assertEqual(int.from_bytes(self.serial.registers[56:58], "little"), 2105)

    def test_cli_timeout_sends_hold_and_reports_failure(self) -> None:
        """관찰 시간이 끝나면 위치 유지 명령을 시도하고 실패 종료 상태를 돌려준다."""
        output = StringIO()
        with patch("scripts.test_motor._observe_move", return_value=False), redirect_stdout(output), redirect_stderr(output):
            result = main(["move", "--port", "FAKE", "--id", "1", "--delta-deg", "5"])
        self.assertEqual(result, 1)
        self.assertIn("정지 명령 전송", output.getvalue())
        self.assertEqual(self.serial.registers[40], 1)


class MotorSetupTests(unittest.TestCase):
    """단회전 설정이 필요한 레지스터만 바꾸고 반복 실행과 실패를 처리하는지 확인한다."""

    def setUp(self) -> None:
        """멀티턴 설정과 영점 보정값이 있는 가짜 모터를 준비한다."""
        self.serial = FakeSerial()
        self.serial.registers[11:13] = b"\x00\x00"
        self.serial.registers[18] = 0xBC
        self.serial.registers[31:33] = (1428).to_bytes(2, "little")
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)

    def test_only_maximum_and_multiturn_bit_change(self) -> None:
        """ID·영점·다른 Phase 비트·토크·목표 위치를 유지하며 단회전으로 바꾼다."""
        before = bytes(self.serial.registers)
        self.assertTrue(configure_single_turn(self.bus, 1))
        changed = [index for index, pair in enumerate(zip(before, self.serial.registers)) if pair[0] != pair[1]]
        self.assertEqual(changed, [11, 12, 18])
        self.assertEqual(self.serial.registers[11:13], bytes.fromhex("ff 0f"))
        self.assertEqual(self.serial.registers[18], 0xAC)
        writes = [packet[5] for packet in self.serial.packets if packet[4] == 3]
        self.assertEqual(writes, [55, 11, 18, 55])
        self.bus._check_position_target(1, 844)
        self.serial.packets.clear()
        self.assertFalse(configure_single_turn(self.bus, 1))
        self.assertTrue(all(packet[4] == 2 for packet in self.serial.packets))

    def test_torque_or_unrelated_configuration_never_writes(self) -> None:
        """토크가 켜졌거나 별도로 지정한 모드·범위가 있으면 쓰기 전에 중단한다."""
        for address, value in [(40, 1), (33, 1), (9, 1), (11, 100)]:
            with self.subTest(address=address):
                before = self.serial.registers[address]
                self.serial.registers[address] = value
                with self.assertRaises(ValueError):
                    configure_single_turn(self.bus, 1)
                self.serial.registers[address] = before
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_failed_write_relocks_and_next_run_completes(self) -> None:
        """최대 위치 쓰기가 반영되지 않아도 잠금을 복구하고 재실행 시 남은 설정만 바꾼다."""
        original_call = self.bus._call

        def ignore_maximum(servo_id: int, method: str, *args: int) -> list[int]:
            """최대 위치 쓰기만 무시해 저장값 검증 실패를 재현한다."""
            if method == "write2ByteTxRx":
                return []
            return original_call(servo_id, method, *args)

        with patch.object(self.bus, "_call", side_effect=ignore_maximum):
            with self.assertRaisesRegex(MotorError, "읽기 검증"):
                configure_single_turn(self.bus, 1)
        self.assertEqual(self.serial.registers[55], 1)
        self.assertTrue(configure_single_turn(self.bus, 1))
        self.assertEqual(self.serial.registers[55], 1)

    def test_explicit_id_uses_saved_port_without_search(self) -> None:
        """ID를 지정하면 저장된 포트의 해당 모터만 설정하고 전체 ID를 검색하지 않는다."""
        with patch("hardware.motor_setup.resolve_port_settings", return_value=PortSettings("FAKE")), \
                patch("hardware.motor_setup.find_motor_id", side_effect=AssertionError("검색하면 안 됩니다")), \
                redirect_stdout(StringIO()):
            self.assertEqual(motor_setup_main(["--id", "1"]), 0)
        self.assertTrue(all(packet[2] == 1 for packet in self.serial.packets))


class MotorIDTests(unittest.TestCase):
    """주소 변경 전후의 실제 SDK 응답 처리와 설정 잠금 복구를 확인한다."""

    def setUp(self) -> None:
        """모터 ID에 따라 응답하는 가짜 포트로 초기 설정 통신을 준비한다."""
        self.serial = FakeSerial()
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.serial)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)

    def test_change_id_accepts_old_new_or_missing_write_ack(self) -> None:
        """ID 변경 응답의 주소가 달라지거나 응답이 없어도 새 주소의 읽기 결과로 확인한다."""
        for reply in ("old", "new", "none"):
            with self.subTest(reply=reply):
                self.serial.registers[5] = 1
                self.serial.registers[40] = 1
                self.serial.id_reply = reply
                before = bytes(self.serial.registers)
                self.assertEqual(change_motor_id(self.bus, 1, 7), 7)
                self.assertEqual(self.serial.registers[5], 7)
                self.assertEqual(self.serial.registers[55], 1)
                changed_addresses = [index for index, (old, new) in enumerate(zip(before, self.serial.registers)) if old != new]
                self.assertEqual(changed_addresses, [5])
        id_writes = [packet for packet in self.serial.packets if packet[4:6] == bytes([3, 5])]
        self.assertEqual(len(id_writes), 3)
        self.assertEqual(id_writes[0], bytes.fromhex("ff ff 01 04 03 05 07 eb"))

    def test_occupied_id_and_corrupt_reply_never_unlock(self) -> None:
        """사용 중인 새 ID와 통신 이상을 모두 감지하고 설정 쓰기 전에 중단한다."""
        original_ping = self.bus.ping
        for failure in (None, COMM_RX_CORRUPT):
            with self.subTest(failure=failure):
                def ping(servo_id: int) -> int:
                    """새 주소에 다른 모터 응답 또는 손상된 응답을 주입한다."""
                    if servo_id == 7:
                        if failure is not None:
                            raise MotorError("손상된 응답", communication_result=failure)
                        return 1234
                    return original_ping(servo_id)

                with patch.object(self.bus, "ping", side_effect=ping):
                    with self.assertRaises((MotorError, ValueError)):
                        change_motor_id(self.bus, 1, 7)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_failed_id_write_relocks_old_id_without_retry(self) -> None:
        """ID 쓰기가 적용되지 않으면 기존 ID를 다시 잠그고 변경 성공으로 보고하지 않는다."""
        self.serial.ignore_id_write = True
        with self.assertRaisesRegex(MotorError, "ID 1 응답과 설정 잠금"):
            change_motor_id(self.bus, 1, 7)
        self.assertEqual(self.serial.registers[5], 1)
        self.assertEqual(self.serial.registers[55], 1)
        self.assertEqual(sum(packet[4:6] == bytes([3, 5]) for packet in self.serial.packets), 1)

    def test_unconfirmed_lock_reports_partial_change(self) -> None:
        """주소가 바뀌어도 잠금을 확인하지 못하면 불완전한 설정으로 보고한다."""
        self.serial.ignore_lock_write = True
        with self.assertRaisesRegex(MotorError, "설정 잠금을 확인하지 못했습니다"):
            change_motor_id(self.bus, 1, 7)
        self.assertEqual(self.serial.registers[5], 7)
        self.assertEqual(self.serial.registers[55], 0)

    def test_same_id_and_invalid_ids_do_not_write(self) -> None:
        """같은 ID는 읽기로 확인하며 범위 밖 ID나 전체 모터 주소는 쓰지 않는다."""
        self.assertEqual(change_motor_id(self.bus, 1, 1), 1)
        for new_id in (-1, 254, 255, True, 1.5):
            with self.subTest(new_id=new_id), self.assertRaises(ValueError):
                change_motor_id(self.bus, 1, new_id)
        self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))

    def test_cli_uses_saved_port(self) -> None:
        """포트 인자를 생략한 ID 설정이 저장된 연결 설정을 사용한다."""
        with patch("hardware.ports.load_port_settings", return_value=PortSettings("FAKE")):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(motor_ids_main(["--new-id", "7"]), 0)
        self.assertEqual(self.serial.registers[5], 7)

    def test_interactive_cli_shows_actual_id_before_accepting_target(self) -> None:
        """이미 변경된 ID를 자동으로 찾아 먼저 보여주고 유효한 목표 입력 후에만 변경한다."""
        self.serial.registers[5] = 7
        output = StringIO()
        answers = iter(["abc", "254", "9"])

        def enter_target(prompt: str) -> str:
            """입력을 요청하는 시점까지 현재 ID 출력과 읽기 전용 상태를 확인한다."""
            self.assertIn("현재 모터 ID: 7", output.getvalue())
            self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))
            return next(answers)

        with patch("hardware.ports.load_port_settings", return_value=PortSettings("FAKE")):
            with patch("hardware.ports.comports", side_effect=AssertionError("전체 포트 검색 금지")):
                with patch("builtins.input", side_effect=enter_target), redirect_stdout(output), redirect_stderr(output):
                    self.assertEqual(motor_ids_main([]), 0)
        self.assertEqual(self.serial.registers[5], 9)
        self.assertIn("ID 확인 완료: 7 → 9", output.getvalue())

    def test_discovery_stops_at_first_reply_and_includes_edge_ids(self) -> None:
        """응답을 찾은 뒤 검색을 멈추며 마지막 ID와 영번 ID도 찾을 수 있다."""
        for target_id in (1, 253, 0):
            with self.subTest(target_id=target_id):
                def read_id(servo_id: int, address: int) -> int:
                    """검색 대상 하나만 응답하게 하여 조회한 ID 순서를 검사한다."""
                    if servo_id == target_id:
                        return target_id
                    raise MotorError("응답 없음", communication_result=COMM_RX_TIMEOUT)

                with patch.object(self.bus, "read_register", side_effect=read_id) as read:
                    self.assertEqual(find_motor_id(self.bus), target_id)
                expected_ids = list(range(1, target_id + 1)) if target_id else [*range(1, 254), 0]
                self.assertEqual([call.args[0] for call in read.call_args_list], expected_ids)

    def test_interactive_cancel_does_not_write(self) -> None:
        """빈 입력과 입력 종료는 모터 설정을 바꾸지 않고 포트를 닫는다."""
        for entered in ("", EOFError()):
            with self.subTest(entered=entered):
                with patch("hardware.ports.load_port_settings", return_value=PortSettings("FAKE")):
                    with patch("builtins.input", side_effect=[entered]), redirect_stdout(StringIO()):
                        self.assertEqual(motor_ids_main([]), 0)
                self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))
                self.assertTrue(self.serial.closed)

    def test_missing_motor_or_bad_response_never_prompts_or_writes(self) -> None:
        """모터를 찾지 못하거나 응답이 손상되면 목표 ID 입력과 설정 쓰기를 시작하지 않는다."""
        for result in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
            with self.subTest(result=result):
                failure = MotorError("검색 실패", communication_result=result)
                with patch("hardware.ports.load_port_settings", return_value=PortSettings("FAKE")):
                    with patch.object(STS3215Bus, "read_register", side_effect=failure):
                        with patch("builtins.input") as prompt, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                            self.assertEqual(motor_ids_main([]), 1)
                        prompt.assert_not_called()
                self.assertFalse(any(packet[4] == 3 for packet in self.serial.packets))


if __name__ == "__main__":
    unittest.main()
