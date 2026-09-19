"""가짜 모터 버스로 1단계 API·공통 상태 스트림·도착 및 정지 연결을 검증한다."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gui.server import Console, create_app
from gui.stage1 import Stage1Session
from hardware.calibration import MotorCalibration
from hardware.joint_control import JointController
from hardware.sts3215 import MotorError, STS3215Bus
from kinematics.joints import ARM_JOINT_NAMES
from test_calibration import FakeMotorChain


class Stage1ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """실제 SDK와 서버를 사용하되 직렬 장치만 가짜로 대체한다."""

    async def asyncSetUp(self) -> None:
        """임시 시작 자세와 로그를 준비하고 실제 관절 제어기를 연결한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.pose_path = root / "stage1.json"
        self.pose_path.write_text(json.dumps({
            "format": "xs.observation_pose.v1", "stage": 1, "fixed_tip": "tip_L",
            "q_rad": [.05, -.5, .03, 1.2, .02, .8, .3],
        }))
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", root / "logs")
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.chain = FakeMotorChain()
        for registers in self.chain.devices.values():
            registers[56:58] = (2048).to_bytes(2, "little")
        serial_patch = patch("scservo_sdk.port_handler.serial.Serial", return_value=self.chain)
        serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.bus = STS3215Bus("FAKE").open()
        self.addCleanup(self.bus.close)
        self.controller = JointController(self.bus)
        self.console = Console(self.controller, MotorCalibration(self.bus))
        self.console.stage1 = Stage1Session(self.console, pose_path=self.pose_path)
        self.session = self.console.stage1
        self.addAsyncCleanup(self.console.close)

    async def _request(self, path: str, method: str = "POST") -> tuple[int, dict]:
        """네트워크 연결 없이 등록된 ASGI 경로를 호출한다."""
        app = create_app("FAKE", None)
        app.state.console = self.console
        messages = []

        async def receive():
            """빈 요청 본문을 전달한다."""
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message):
            """실제 경로의 응답을 수집한다."""
            messages.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
                 "query_string": b"", "headers": [(b"content-type", b"application/json")],
                 "server": ("test", 80), "client": ("test", 1), "root_path": ""}
        await app(scope, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        return status, json.loads(b"".join(message.get("body", b"") for message in messages))

    async def test_start_stream_and_arrival_preserve_grippers(self) -> None:
        """몸통만 한 번 전송하고 공통 스트림과 조회 API에 같은 도착 신호가 나타난다."""
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        status, receipt = await self._request("/api/stage1/start")
        self.assertEqual(status, 200)
        self.assertEqual(receipt["state"], "MOVING")
        self.assertIsNone(receipt["arrival"])
        packets = [packet for packet in self.chain.packets if packet[4] == 0x83]
        self.assertEqual(len(packets), 1)
        self.assertEqual(list(packets[0][7:-1:8]), list(range(1, 8)))
        async with self.console.subscribe() as updates:
            self.console.start()
            payload = await asyncio.wait_for(updates.get(), timeout=2.)
        self.assertEqual(payload["stage1"]["state"], "ARRIVED")
        status, current = await self._request("/api/stage1", "GET")
        self.assertEqual(current["arrival"]["run_id"], receipt["run_id"])
        self.assertEqual(current["arrival"], payload["stage1"]["arrival"])
        self.assertTrue(all(state["arrived_now"] for state in payload["joints"] if state["name"] in ARM_JOINT_NAMES))
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})

    async def test_current_position_is_rechecked_after_latched_arrival(self) -> None:
        """기존 제어기에 도착이 기록된 뒤 위치가 변하면 단계 완료를 기다린다."""
        await self.session.start()
        await self.console.snapshot()
        registers = self.chain.devices[1]
        target = int.from_bytes(registers[42:44], "little")
        registers[56:58] = (target + 6).to_bytes(2, "little")
        states = await self.console.snapshot()
        joint = next(state for state in states if state["name"] == "J1")
        self.assertEqual(joint["motion_status"], "arrived")
        self.assertFalse(joint["arrived_now"])
        await self.session.observe(states)
        self.assertEqual(self.session.status()["state"], "MOVING")
        registers[56:58] = target.to_bytes(2, "little")
        await self.session.observe(await self.console.snapshot())
        self.assertEqual(self.session.status()["state"], "ARRIVED")

    async def test_joint_timeout_preserves_moving_targets_until_current_arrival(self) -> None:
        """J2의 미세 오차와 시간 초과가 이동 중인 J4 목표를 정지 명령으로 바꾸지 않는다."""
        await self.session.start()
        command_ids = dict(self.session.tracker.command_ids)
        goals = {index: bytes(self.chain.devices[index][42:44]) for index in range(1, 8)}
        self.chain.devices[2][56:58] = (int.from_bytes(goals[2], "little") - 4).to_bytes(2, "little")
        self.chain.devices[4][56:58] = (int.from_bytes(goals[4], "little") + 300).to_bytes(2, "little")
        self.chain.devices[4][58:60] = (100).to_bytes(2, "little")
        self.controller._targets["J2"].deadline = 0.
        states = await self.console.snapshot()
        self.assertEqual(next(state for state in states if state["name"] == "J2")["motion_status"], "timeout")
        self.assertFalse(next(state for state in states if state["name"] == "J2")["arrived_now"])
        self.assertEqual(next(state for state in states if state["name"] == "J4")["motion_status"], "moving")
        await self.session.observe(states)
        self.assertEqual(self.session.status()["state"], "MOVING")
        self.assertIsNone(self.session.status()["arrival"])
        self.assertEqual({name: self.controller.command_id(name) for name in ARM_JOINT_NAMES}, command_ids)
        self.assertEqual({index: bytes(self.chain.devices[index][42:44]) for index in range(1, 8)}, goals)
        for index in range(1, 8):
            self.chain.devices[index][56:58] = goals[index]
            self.chain.devices[index][58:60] = bytes(2)
        states = await self.console.snapshot()
        self.assertEqual(next(state for state in states if state["name"] == "J2")["motion_status"], "timeout")
        await self.session.observe(states)
        self.assertEqual(self.session.status()["state"], "ARRIVED")

    async def test_stage_timeout_still_stops_owned_targets(self) -> None:
        """개별 예상시간과 별개인 단계 전체 기한이 지나면 몸통 명령을 정지한다."""
        await self.session.start()
        command_ids = dict(self.session.tracker.command_ids)
        grippers = {index: bytes(self.chain.devices[index]) for index in (0, 8)}
        self.session.tracker._clock = lambda: self.session.tracker.started_at_s + self.session.tracker.timeout_s
        await self.session.observe(await self.console.snapshot())
        self.assertEqual(self.session.status()["state"], "FAILED")
        self.assertIsNone(self.session.status()["arrival"])
        self.assertTrue(all(self.controller.command_id(name) != command_ids[name] for name in ARM_JOINT_NAMES))
        self.assertEqual(grippers, {index: bytes(self.chain.devices[index]) for index in (0, 8)})

    async def test_manual_replacement_is_preserved_while_remaining_stage_targets_stop(self) -> None:
        """다른 조작으로 바뀐 관절 명령을 덮어쓰지 않고 나머지 단계 명령만 정지한다."""
        await self.session.start()
        original_ids = dict(self.session.tracker.command_ids)
        await self.console.move("J1", 8., 10.)
        replacement_id = self.controller.command_id("J1")
        target = bytes(self.chain.devices[1][42:44])
        await self.session.observe(await self.console.snapshot())
        self.assertEqual(self.session.status()["state"], "STOPPED")
        self.assertIsNone(self.session.status()["arrival"])
        self.assertEqual(self.controller.command_id("J1"), replacement_id)
        self.assertEqual(bytes(self.chain.devices[1][42:44]), target)
        for name in ARM_JOINT_NAMES[1:]:
            self.assertNotEqual(self.controller.command_id(name), original_ids[name])

    async def test_duplicate_start_and_invalid_file_do_not_send_again(self) -> None:
        """동시 시작과 잘못된 저장 자세로 추가 이동 명령이 나가지 않는다."""
        results = await asyncio.gather(self.session.start(), self.session.start(), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, MotorError) for result in results), 1)
        self.assertEqual(len([packet for packet in self.chain.packets if packet[4] == 0x83]), 1)
        await self.session.stop()
        self.pose_path.write_text('{"format": "wrong"}')
        status, _ = await self._request("/api/stage1/start")
        self.assertEqual(status, 400)
        self.assertEqual(len([packet for packet in self.chain.packets if packet[4] == 0x83]), 1)

    async def test_command_after_snapshot_cannot_emit_stale_arrival(self) -> None:
        """도착 표본을 읽은 직후 다른 명령이 들어오면 이전 표본으로 완료하지 않는다."""
        await self.session.start()
        states = await self.console.snapshot()
        self.assertTrue(all(state["arrived_now"] for state in states if state["name"] in ARM_JOINT_NAMES))
        await self.console.move("J1", 8., 10.)
        replacement_id = self.controller.command_id("J1")
        await self.session.observe(states)
        self.assertEqual(self.session.status()["state"], "STOPPED")
        self.assertIsNone(self.session.status()["arrival"])
        self.assertEqual(self.controller.command_id("J1"), replacement_id)

    async def test_cancelled_request_keeps_registration_and_stop_api_works(self) -> None:
        """요청자가 떠나도 전송 후 추적 등록을 끝내고 정지 API가 해당 실행을 멈춘다."""
        started, release = asyncio.Event(), asyncio.Event()
        original = self.console.move_pose

        async def delayed(*args, **kwargs):
            """전송 이후 응답 전에 요청 취소가 발생하는 상황을 재현한다."""
            receipt = await original(*args, **kwargs)
            started.set()
            await release.wait()
            return receipt

        with patch.object(self.console, "move_pose", side_effect=delayed):
            request = asyncio.create_task(self.session.start())
            await asyncio.wait_for(started.wait(), timeout=2.)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            release.set()
            await self.session._start_task
        self.assertEqual(self.session.status()["state"], "MOVING")
        status, stopped = await self._request("/api/stage1/stop")
        self.assertEqual(status, 200)
        self.assertEqual(stopped["state"], "STOPPED")
        self.assertIsNone(stopped["arrival"])


if __name__ == "__main__":
    unittest.main()
