"""가짜 관절 제어기로 GUI 서버의 오류 격리와 상태 공유 및 통신 실행 순서를 검증한다."""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from fastapi import WebSocketDisconnect

from gui.server import Console, create_app
from hardware.calibration import MotorCalibration
from hardware.joint_control import DEFAULT_CALIBRATION_PATH, JointController, JointState
from hardware.sts3215 import MotorError, STS3215Bus
from test_calibration import FakeMotorChain


class ConsoleTests(unittest.IsolatedAsyncioTestCase):
    """실물 연결 없이 서버의 정상 처리와 부분 실패를 확인한다."""

    async def asyncSetUp(self) -> None:
        """가짜 모터 아홉 개와 실제 콘솔 실행부를 준비한다."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        log_patch = patch("hardware.motor_logging.DEFAULT_LOG_DIR", Path(directory.name) / "logs")
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.pose_path = Path(directory.name) / "last_pose.json"
        pose_patch = patch("gui.server.LAST_POSE_PATH", self.pose_path)
        pose_patch.start()
        self.addCleanup(pose_patch.stop)
        self.names = ("G_L", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "G_R")
        self.controller = Mock(spec=JointController)
        self.controller.joint_names = self.names
        self.controller.read.side_effect = lambda name: JointState(name, 12.0, 0.0)
        self.controller.read_torque.return_value = True
        self.calibration = Mock(spec=MotorCalibration)
        self.calibration.calibrate_all_zero.return_value = self.names
        self.console = Console(self.controller, self.calibration)
        self.addAsyncCleanup(self.console.close)

    async def _post(self, path: str, body: dict, *, method: str = "POST") -> tuple[int, dict]:
        """네트워크와 실물 포트를 열지 않고 실제 HTTP 경로를 실행한다."""
        app = create_app("FAKE", None)
        app.state.console = self.console
        messages = []

        async def receive() -> dict:
            """요청 본문을 ASGI 앱에 전달한다."""
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

        async def send(message: dict) -> None:
            """앱의 응답을 수집한다."""
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [(b"content-type", b"application/json")],
            "server": ("test", 80), "client": ("test", 1), "root_path": "",
        }
        await app(scope, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        payload = b"".join(message.get("body", b"") for message in messages)
        return status, json.loads(payload)

    async def test_recent_pose_survives_reset_gripper_and_console_restart(self) -> None:
        """실제 측정값 대신 입력한 목표를 보존하며 초기화와 그리퍼 명령으로 덮어쓰지 않는다."""
        targets = {name: 7.13 for name in self.names if name.startswith("J")}
        self.controller.move_many.side_effect = lambda angles: dict.fromkeys(angles, 7.119)
        self.controller.command_id.return_value = 1
        status, _ = await self._post("/api/pose", {"angles_deg": targets})
        self.assertEqual(status, 200)
        status, _ = await self._post("/api/pose", {
            "angles_deg": dict.fromkeys(targets, 0.0), "remember": False,
        })
        self.assertEqual(status, 200)
        status, _ = await self._post("/api/pose", {"angles_deg": {"G_L": 4.6}})
        self.assertEqual(status, 200)
        restarted = Console(self.controller, self.calibration)
        self.addAsyncCleanup(restarted.close)
        self.assertEqual(await restarted.last_pose(), {"angles_deg": targets})
        calls_before = self.controller.move_many.call_count
        status, payload = await self._post("/api/pose/last", {}, method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(payload["pose"]["angles_deg"], targets)
        self.assertEqual(self.controller.move_many.call_count, calls_before)

    async def test_failed_motion_preserves_previous_pose_and_save_failure_reports_acceptance(self) -> None:
        """거부된 목표를 저장하지 않고 파일 저장 실패와 이미 전송된 이동을 구분한다."""
        original = {"angles_deg": {name: 2.0 for name in self.names if name.startswith("J")}}
        self.pose_path.write_text(json.dumps(original))
        targets = dict.fromkeys(original["angles_deg"], 5.0)
        self.controller.move_many.side_effect = ValueError("관절 범위 초과")
        status, _ = await self._post("/api/pose", {"angles_deg": targets})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(self.pose_path.read_text()), original)
        self.controller.move_many.side_effect = None
        self.controller.move_many.return_value = targets
        self.controller.command_id.return_value = 1
        with patch("pathlib.Path.replace", side_effect=OSError("쓰기 실패")):
            status, payload = await self._post("/api/pose", {"angles_deg": targets})
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "accepted")
        self.assertIn("쓰기 실패", payload["pose_save_error"])
        self.assertEqual(json.loads(self.pose_path.read_text()), original)

    async def test_recent_pose_missing_or_invalid_file(self) -> None:
        """기록이 없거나 손상된 경우 임의의 자세를 만들지 않고 원인을 알린다."""
        status, payload = await self._post("/api/pose/last", {}, method="GET")
        self.assertEqual(status, 200)
        self.assertIsNone(payload["pose"])
        self.pose_path.write_text('{"angles_deg": {"J1": 5}}')
        status, _ = await self._post("/api/pose/last", {}, method="GET")
        self.assertEqual(status, 400)
        self.pose_path.write_text("broken")
        status, _ = await self._post("/api/pose/last", {}, method="GET")
        self.assertEqual(status, 400)
        self.controller.move_many.assert_not_called()

    async def test_gain_endpoint_uses_shared_controller_and_rejects_gripper(self) -> None:
        """게인 변경은 기존 제어기에 전달하고 그리퍼·잘못된 값·촬영 중 요청은 거부한다."""
        self.controller.set_position_gains.return_value = {"after": {"pid_p": 40, "pid_i": 0, "pid_d": 32}}
        request = {"joint": "J2", "p": 40, "i": 0, "d": 32}
        status, payload = await self._post("/api/gains", request)
        self.assertEqual(status, 200)
        self.assertEqual(payload["after"]["pid_p"], 40)
        self.controller.set_position_gains.assert_called_once_with("J2", p=40, i=0, d=32)
        status, _ = await self._post("/api/gains", {**request, "joint": "G_L"})
        self.assertEqual(status, 400)
        status, _ = await self._post("/api/gains", {**request, "p": True})
        self.assertEqual(status, 422)
        self.console._scan_task = asyncio.create_task(asyncio.Event().wait())
        try:
            status, _ = await self._post("/api/gains", request)
            self.assertEqual(status, 400)
        finally:
            self.console._scan_task.cancel()
            await asyncio.gather(self.console._scan_task, return_exceptions=True)
        self.assertEqual(self.controller.set_position_gains.call_count, 1)

    async def test_zero_endpoints_use_calibration_and_refresh_controller(self) -> None:
        """두 영점 경로가 설정 모듈에 위임하고 제어기의 기준을 갱신한다."""
        status, _ = await self._post("/api/zero", {"joint": "J1"})
        self.assertEqual(status, 200)
        self.calibration.save_zero.assert_called_once_with("J1")
        self.controller.reload_calibration.assert_called_once_with()
        status, payload = await self._post("/api/calibrate-zero", {})
        self.assertEqual(status, 200)
        self.assertEqual(payload["joints"], list(self.names))
        self.calibration.calibrate_all_zero.assert_called_once_with()
        self.assertEqual(self.controller.reload_calibration.call_count, 2)

    async def test_scan_keeps_polling_and_stop_available_after_disconnect(self) -> None:
        """촬영 중 조회와 정지가 계속되며 요청 취소 후에도 중복 촬영 없이 저장한다."""
        started, release = threading.Event(), threading.Event()
        self.controller.log_path = Path("fake-motors.jsonl")

        def capture(read_pose, log_path) -> dict:
            """별도 카메라 스레드에서 모터 실행부를 통해 실제 자세를 조회한다."""
            self.assertEqual(read_pose()["angles_deg"], dict.fromkeys(self.names, 12.0))
            self.assertEqual(log_path, self.controller.log_path)
            started.set()
            release.wait(2)
            read_pose()
            return {"path": "/fake/scan", "frames": 15}

        self.console.start()
        with patch("gui.server.capture_scan", side_effect=capture):
            request = asyncio.create_task(self.console.scan())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                reads = self.controller.read.call_count
                async with self.console.subscribe() as updates:
                    payload = await asyncio.wait_for(updates.get(), 1)
                    self.assertEqual(payload["scan"]["status"], "running")
                self.assertGreater(self.controller.read.call_count, reads)
                status, _ = await self._post("/api/pose", {"angles_deg": {"J1": 5}})
                self.assertEqual(status, 400)
                status, _ = await self._post("/api/scan", {})
                self.assertEqual(status, 400)
                status, _ = await self._post("/api/stop", {"joint": "J1"})
                self.assertEqual(status, 200)
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
            finally:
                release.set()
                await asyncio.wait_for(asyncio.shield(self.console._scan_task), 1)
        self.assertEqual(self.console.scan_status["status"], "saved")
        self.controller.move_many.assert_not_called()

    async def test_scan_permission_error_is_visible_and_retryable(self) -> None:
        """USB 권한 실패를 GUI에 설명하고 다음 요청을 막지 않는다."""
        from hardware.camera import CameraError

        with patch("gui.server.capture_scan", side_effect=CameraError("uvc_open failed -3")):
            status, payload = await self._post("/api/scan", {})
        self.assertEqual(status, 400)
        self.assertIn("sudo .venv/bin/python -m gui.server", payload["detail"])
        self.assertEqual(self.console.scan_status["status"], "error")
        with patch("gui.server.capture_scan", return_value={"path": "/fake/scan", "frames": 15}):
            status, payload = await self._post("/api/scan", {})
        self.assertEqual(status, 200)
        self.assertEqual(payload["frames"], 15)

    async def test_partial_calibration_failure_refreshes_controller(self) -> None:
        """중점 설정 실패도 제어기에 반영하고 성공 응답을 보내지 않는다."""
        self.calibration.calibrate_all_zero.side_effect = MotorError("중점 설정 미완료")
        status, payload = await self._post("/api/calibrate-zero", {})
        self.assertEqual(status, 400)
        self.assertIn("미완료", payload["detail"])
        self.controller.reload_calibration.assert_called_once_with()

    async def test_pose_endpoint_uses_shared_bus_and_returns_matching_state_ids(self) -> None:
        """묶음 API가 실제 SDK 동기 패킷과 상태·raw 오차로 이어지는지 확인한다."""
        serial = FakeMotorChain()
        original_console = self.console
        with patch("scservo_sdk.port_handler.serial.Serial", return_value=serial):
            with STS3215Bus("FAKE") as bus:
                controller = JointController(bus)
                self.console = Console(controller, MotorCalibration(bus))
                try:
                    angles = {f"J{index}": 5.0 for index in range(1, 8)}
                    status, payload = await self._post("/api/pose", {"angles_deg": angles})
                    self.assertEqual(status, 200)
                    self.assertEqual(payload["status"], "accepted")
                    self.assertEqual(set(payload["targets_deg"]), set(angles))
                    states = await self.console.snapshot()
                    for state in states:
                        if state["name"] in angles:
                            self.assertEqual(state["command_id"], payload["command_ids"][state["name"]])
                            self.assertEqual(state["motion_status"], "arrived")
                            self.assertEqual(state["error_raw"], 0)
                    self.assertEqual(sum(packet[4] == 0x83 for packet in serial.packets), 1)
                    self.assertEqual(serial.devices[0][40], 0)
                    self.assertEqual(serial.devices[8][40], 0)
                    count = len(serial.packets)
                    status, _ = await self._post("/api/pose", {"angles_deg": {"J1": True}})
                    self.assertEqual(status, 422)
                    self.assertEqual(len(serial.packets), count)
                finally:
                    await self.console.close()
                    self.console = original_console

    async def test_real_controller_arrival_reaches_gui_for_absolute_and_relative_moves(self) -> None:
        """단일·전체 이동이 실제 제어기와 SDK를 거쳐 도착 판정과 로그에 반영된다."""
        serial = FakeMotorChain()
        original_console = self.console
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            path.write_bytes(DEFAULT_CALIBRATION_PATH.read_bytes())
            saved_calibration = path.read_bytes()
            with patch("scservo_sdk.port_handler.serial.Serial", return_value=serial):
                with STS3215Bus("FAKE") as bus:
                    controller = JointController(bus, path)
                    self.console = Console(controller, MotorCalibration(bus, path))
                    try:
                        for route, body in (("/api/move", {"joint": "J1", "angle_deg": 5}),
                                            ("/api/jog", {"joint": "J1", "delta_deg": 5}),
                                            ("/api/move-zero", {"speed_deg_s": 8})):
                            status, payload = await self._post(route, body)
                            self.assertEqual(status, 200)
                            self.assertEqual(payload["status"], "accepted")
                            if route == "/api/move-zero":
                                for device in serial.devices.values():
                                    self.assertEqual(int.from_bytes(device[42:44], "little"), 2048)
                                    self.assertEqual(int.from_bytes(device[46:48], "little"), 91)
                                self.assertEqual(path.read_bytes(), saved_calibration)
                            target = int.from_bytes(serial.devices[1][42:44], "little")
                            for offset, expected in ((4, "moving"), (3, "arrived")):
                                serial.devices[1][56:58] = (target + offset).to_bytes(2, "little")
                                states = await self.console.snapshot()
                                joint = next(item for item in states if item["name"] == "J1")
                                self.assertEqual(joint["motion_status"], expected)
                                self.assertEqual(joint["tolerance_deg"], controller.tolerance_for("J1"))
                                self.assertIsNone(joint["error"])
                        rows = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
                        self.assertEqual(sum(row["event"] == "command_result" for row in rows), 11)
                        zero_commands = [row for row in rows if row["event"] == "command" and row.get("angle_deg") == 0]
                        self.assertEqual({row["joint"] for row in zero_commands}, set(self.names))
                        samples = [row for row in rows if row["event"] == "sample"]
                        self.assertEqual({row["joint"] for row in samples}, set(self.names))
                        self.assertTrue(all("current_ma" in row and "pwm_percent" in row for row in samples))
                    finally:
                        await self.console.close()
                        self.console = original_console

    async def test_calibration_survives_disconnect_and_rejects_new_commands(self) -> None:
        """요청 취소 후에도 설정을 마치며 진행 중 이동과 중복 설정을 거부한다."""
        started, release = threading.Event(), threading.Event()

        def calibrate() -> tuple[str, ...]:
            """진행 중인 중점 설정을 흉내 낸다."""
            started.set()
            release.wait(2)
            return self.names

        self.calibration.calibrate_all_zero.side_effect = calibrate
        request = asyncio.create_task(self.console.calibrate_all_zero())
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        try:
            for path, body in (("/api/move", {"joint": "J1", "angle_deg": 5}),
                               ("/api/pose", {"angles_deg": {"J1": 5}}),
                               ("/api/move-zero", {}),
                               ("/api/torque", {"enabled": True}),
                               ("/api/calibrate-zero", {})):
                status, _ = await self._post(path, body)
                self.assertEqual(status, 400)
        finally:
            release.set()
        await asyncio.wait_for(asyncio.shield(self.console._calibration_task), 1)
        self.calibration.calibrate_all_zero.assert_called_once_with()
        self.controller.reload_calibration.assert_called_once_with()
        self.controller.move_to.assert_not_called()

    async def test_all_zero_reports_failed_joints_without_changing_calibration(self) -> None:
        """전체 영점 이동의 일부 실패를 알리고 영점 재설정은 호출하지 않는다."""
        def move(name: str, angle_deg: float, *, speed_deg_s: float) -> None:
            """두 관절에만 통신 오류를 주입한다."""
            if name in ("J2", "J6"):
                raise MotorError("응답 실패")

        self.controller.move_to.side_effect = move
        status, payload = await self._post("/api/move-zero", {"speed_deg_s": 8})
        self.assertEqual(status, 400)
        self.assertIn("J2", payload["detail"])
        self.assertIn("J6", payload["detail"])
        self.assertEqual(self.controller.move_to.call_count, 9)
        self.calibration.save_zero.assert_not_called()
        self.calibration.calibrate_all_zero.assert_not_called()
        self.controller.reload_calibration.assert_not_called()

    async def test_all_stop_attempts_every_joint_and_reports_failures(self) -> None:
        """전체 정지에서 여러 관절이 실패해도 끝까지 시도하고 실패 대상을 반환한다."""
        def stop(name: str) -> None:
            """첫 관절과 중간 관절에 정지 오류를 주입한다."""
            if name in ("G_L", "J4"):
                raise MotorError("응답 실패")

        self.controller.stop.side_effect = stop
        status, payload = await self._post("/api/stop", {})
        self.assertEqual(status, 400)
        self.assertIn("G_L", payload["detail"])
        self.assertIn("J4", payload["detail"])
        self.assertEqual([call.args[0] for call in self.controller.stop.call_args_list], list(self.names))

    async def test_all_torque_continues_after_error_and_single_target_stays_single(self) -> None:
        """전체 토크 실패를 격리하며 단일 관절 요청의 대상은 확대하지 않는다."""
        def torque(name: str, enabled: bool) -> None:
            """첫 관절의 토크 명령만 실패시킨다."""
            if name == "G_L":
                raise OSError("응답 실패")

        self.controller.set_torque.side_effect = torque
        status, _ = await self._post("/api/torque", {"enabled": False})
        self.assertEqual(status, 400)
        self.assertEqual(self.controller.set_torque.call_count, 9)
        self.controller.set_torque.reset_mock()
        status, _ = await self._post("/api/torque", {"joint": "J1", "enabled": False})
        self.assertEqual(status, 200)
        self.controller.set_torque.assert_called_once_with("J1", False)

    async def test_snapshot_keeps_healthy_joints_and_clears_failed_values(self) -> None:
        """한 관절의 읽기 실패가 다른 관절의 상태를 지우지 않으며 복구 후 다시 읽힌다."""
        def read(name: str) -> JointState:
            """첫 관절에만 읽기 오류를 주입한다."""
            if name == "G_L":
                raise MotorError("응답 실패")
            return JointState(name, 12.0, 0.0)

        self.controller.read.side_effect = read
        states = await self.console.snapshot()
        self.assertEqual(len(states), 9)
        self.assertIsNone(states[0]["position_deg"])
        self.assertIsNone(states[0]["torque"])
        self.assertIn("응답 실패", states[0]["error"])
        self.assertTrue(all(state["error"] is None and state["position_deg"] == 12 for state in states[1:]))
        self.controller.read.side_effect = lambda name: JointState(name, 15.0, 0.0)
        self.assertIsNone((await self.console.snapshot())[0]["error"])

    async def test_torque_read_failure_is_not_displayed_as_torque_off(self) -> None:
        """공통 제어기의 토크 조회 실패를 토크 해제로 표시하지 않는다."""
        def read_torque(name: str) -> bool:
            """첫 관절에만 토크 조회 오류를 주입한다."""
            if name == "G_L":
                raise MotorError("알 수 없는 토크 상태")
            return True

        self.controller.read_torque.side_effect = read_torque
        states = await self.console.snapshot()
        self.assertIsNone(states[0]["torque"])
        self.assertIsNotNone(states[0]["error"])
        self.assertTrue(all(state["torque"] for state in states[1:]))

    async def test_two_subscribers_share_poll_and_recording_continues_without_browser(self) -> None:
        """여러 화면에 조회 결과를 공유하고 화면이 없어도 기록용 조회를 계속한다."""
        with patch("gui.server.POLL_INTERVAL_S", 0.01):
            self.console.start()
            async with self.console.subscribe() as first, self.console.subscribe() as second:
                a, b = await asyncio.wait_for(asyncio.gather(first.get(), second.get()), 1)
                self.assertIs(a, b)
                self.assertEqual(self.controller.read.call_count, 9)
                self.assertEqual(self.controller.read_torque.call_count, 9)
            await asyncio.sleep(0.04)
            self.assertGreater(self.controller.read.call_count, 9)

    async def test_slow_subscriber_receives_latest_snapshot_without_backlog(self) -> None:
        """화면의 수신이 느려도 오래된 상태가 쌓이지 않는다."""
        with patch("gui.server.POLL_INTERVAL_S", 0.01):
            self.console.start()
            async with self.console.subscribe() as slow, self.console.subscribe() as fast:
                await asyncio.wait_for(fast.get(), 1)
                self.controller.read.side_effect = lambda name: JointState(name, 25.0, 0.0)
                latest = await asyncio.wait_for(fast.get(), 1)
                self.assertEqual(slow.qsize(), 1)
                self.assertIs(slow.get_nowait(), latest)
                self.assertEqual(latest["joints"][0]["position_deg"], 25)

    async def test_cancelled_request_cannot_overlap_running_serial_work(self) -> None:
        """요청 취소 뒤에도 첫 통신이 끝나기 전에 다음 통신이 시작되지 않는다."""
        started, release, next_started = threading.Event(), threading.Event(), threading.Event()

        def first_work() -> None:
            """취소 이후에도 실행 중인 통신을 흉내 낸다."""
            started.set()
            release.wait(2)

        first = asyncio.create_task(self.console.run(first_work))
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(self.console.run(next_started.set))
        try:
            await asyncio.sleep(0.02)
            self.assertFalse(next_started.is_set())
        finally:
            release.set()
        await asyncio.wait_for(second, 1)
        self.assertTrue(next_started.is_set())

    async def test_stop_can_run_between_joint_reads(self) -> None:
        """전체 상태 조회 도중 들어온 정지가 남은 관절 조회보다 먼저 실행된다."""
        started, release = threading.Event(), threading.Event()
        events = []

        def read(name: str) -> JointState:
            """첫 관절의 조회를 잠시 잡아 두고 실행 순서를 기록한다."""
            events.append(("read", name))
            if name == "G_L":
                started.set()
                release.wait(2)
            return JointState(name, 0.0, 0.0)

        self.controller.read.side_effect = read
        self.controller.stop.side_effect = lambda name: events.append(("stop", name))
        snapshot = asyncio.create_task(self.console.snapshot())
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        stop = asyncio.create_task(self.console.stop(("J7",)))
        try:
            await asyncio.sleep(0.01)
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(snapshot, stop), 1)
        self.assertLess(events.index(("stop", "J7")), events.index(("read", "J1")))

    async def test_websocket_disconnect_removes_subscription(self) -> None:
        """새 상태를 기다리는 중에도 화면 연결이 끊기면 구독을 정리한다."""
        app = create_app("FAKE", None)
        app.state.console = self.console
        endpoint = next(route.endpoint for route in app.routes if route.path == "/ws")
        websocket = Mock()

        async def accept() -> None:
            """가짜 화면 연결을 수락한다."""

        async def disconnect() -> None:
            """클라이언트 연결 종료를 즉시 알린다."""
            raise WebSocketDisconnect()

        websocket.accept = accept
        websocket.receive_text = disconnect
        await asyncio.wait_for(endpoint(websocket), 1)
        self.assertFalse(self.console._subscribers)
        self.controller.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
