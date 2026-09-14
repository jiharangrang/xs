"""기존 관절 제어기에 화면의 명령을 전달하고 관절별 상태를 공유하는 로컬 서버다.
모터 통신은 전용 스레드 하나에서 수행하며, 한 관절의 오류가 다른 관절 처리를 막지 않는다.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, TypeVar

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from hardware.calibration import MotorCalibration
from hardware.joint_control import JointController
from hardware.ports import resolve_port_settings
from hardware.sts3215 import MotorError, STS3215Bus


INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
POLL_INTERVAL_S = 0.2
BUS_ERRORS = (ValueError, MotorError, OSError)
T = TypeVar("T")


class Console:
    """관절 제어기의 명령과 상태 조회를 한 번에 하나씩 수행한다."""

    def __init__(self, controller: JointController, calibration: MotorCalibration) -> None:
        """관절 제어와 캘리브레이션을 연결하고 통신 전용 스레드를 준비한다."""
        self._controller = controller
        self._calibration = calibration
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motor-console")
        self._subscribers: set[asyncio.Queue] = set()
        self._poll_task: asyncio.Task | None = None
        self._closing = False
        self._calibration_task: asyncio.Task | None = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        """설정 파일에 정의된 관절 순서를 그대로 전달한다."""
        return self._controller.joint_names

    def start(self) -> None:
        """화면 연결과 관계없이 상태 조회와 공통 로그 기록을 시작한다."""
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_states())

    async def close(self) -> None:
        """조회 작업을 종료하고 진행 중인 통신이 끝날 때까지 기다린다."""
        self._closing = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
        await asyncio.to_thread(self._worker.shutdown, wait=True, cancel_futures=True)

    async def run(self, work: Callable[[], T]) -> T:
        """요청이 취소돼도 통신이 겹치지 않도록 전용 스레드에서 순서대로 수행한다."""
        if self._closing:
            raise MotorError("서버가 종료 중입니다.")
        return await asyncio.get_running_loop().run_in_executor(self._worker, work)

    def _read_joint(self, name: str) -> dict:
        """관절 하나의 각도·속도·토크 상태를 읽는다."""
        state = self._controller.read(name)
        torque = state.torque_enabled
        if torque is None:
            torque = self._controller.read_torque(name)
        return {
            "name": name,
            "position_deg": state.position_deg,
            "speed_deg_s": state.speed_deg_s,
            "torque": torque,
            "target_deg": state.target_deg,
            "error_deg": state.error_deg,
            "tolerance_deg": state.tolerance_deg,
            "motion_status": state.motion_status,
            "error": None,
        }

    async def snapshot(self) -> list[dict]:
        """관절마다 읽기 실패를 기록하고 조회 사이에 명령이 실행될 기회를 준다."""
        snapshot = []
        for name in self.joint_names:
            try:
                snapshot.append(await self.run(lambda name=name: self._read_joint(name)))
            except BUS_ERRORS as error:
                snapshot.append({
                    "name": name, "position_deg": None, "speed_deg_s": None,
                    "torque": None, "error": str(error),
                    "target_deg": None, "error_deg": None, "tolerance_deg": None,
                    "motion_status": "unknown",
                })
        return snapshot

    @asynccontextmanager
    async def subscribe(self):
        """공통 조회 결과를 구독하고 연결 종료 시 구독을 해제한다."""
        updates = asyncio.Queue(maxsize=1)
        self._subscribers.add(updates)
        try:
            yield updates
        finally:
            self._subscribers.discard(updates)

    async def _poll_states(self) -> None:
        """상태를 계속 읽어 로그에 남기고 연결된 화면에는 최신 값만 전달한다."""
        while True:
            payload = {"joints": await self.snapshot(), "error": None}
            for updates in self._subscribers:
                if updates.full():
                    updates.get_nowait()
                updates.put_nowait(payload)
            await asyncio.sleep(POLL_INTERVAL_S)

    def _apply_all(self, joints: tuple[str, ...], work: Callable[[str], object]) -> None:
        """대상 관절을 모두 처리한 뒤 실패한 관절만 모아서 알린다."""
        errors = []
        for name in joints:
            try:
                work(name)
            except BUS_ERRORS as error:
                errors.append(f"{name}: {error}")
        if errors:
            raise MotorError("처리하지 못한 관절: " + "; ".join(errors))

    def _require_idle_calibration(self) -> None:
        """영점 변경 중 이전 좌표를 기준으로 새 명령이 대기열에 쌓이지 않게 한다."""
        if self._calibration_task is not None and not self._calibration_task.done():
            raise MotorError("전체 중점·영점 설정 중입니다. 완료 후 명령해 주세요.")

    async def calibrate_all_zero(self) -> tuple[str, ...]:
        """브라우저 요청이 끊겨도 시작한 중점 변경과 영점 저장을 끝까지 처리한다."""
        self._require_idle_calibration()
        self._calibration_task = asyncio.create_task(
            self.run(lambda: self._update_calibration(self._calibration.calibrate_all_zero))
        )
        self._calibration_task.add_done_callback(self._finish_calibration)
        return await asyncio.shield(self._calibration_task)

    def _finish_calibration(self, task: asyncio.Task) -> None:
        """요청자가 연결을 끊었어도 작업의 실패 결과를 회수한다."""
        if not task.cancelled():
            task.exception()

    def _update_calibration(self, work: Callable[[], T]) -> T:
        """설정 작업의 성공 여부와 관계없이 관절 제어기에 최신 저장 상태를 반영한다."""
        try:
            return work()
        finally:
            self._controller.reload_calibration()

    async def move(self, joint: str, angle_deg: float, speed_deg_s: float) -> None:
        """절대 목표 각도를 전송한다."""
        self._require_idle_calibration()
        await self.run(lambda: self._controller.move_to(joint, angle_deg, speed_deg_s=speed_deg_s))

    async def move_all_zero(self, speed_deg_s: float) -> None:
        """모든 관절에 저장된 영점으로의 이동 명령을 차례로 전송한다."""
        self._require_idle_calibration()
        await self.run(lambda: self._apply_all(
            self.joint_names, lambda name: self._controller.move_to(name, 0.0, speed_deg_s=speed_deg_s),
        ))

    async def jog(self, joint: str, delta_deg: float, speed_deg_s: float) -> None:
        """현재 각도 기준 상대 이동을 전송한다."""
        self._require_idle_calibration()
        await self.run(lambda: self._controller.move_by(joint, delta_deg, speed_deg_s=speed_deg_s))

    async def stop(self, joints: tuple[str, ...]) -> None:
        """지정한 관절을 현재 위치에서 멈춘다."""
        await self.run(lambda: self._apply_all(joints, self._controller.stop))

    async def set_torque(self, joints: tuple[str, ...], enabled: bool) -> None:
        """지정한 관절의 토크만 켜거나 끈다."""
        if enabled:
            self._require_idle_calibration()
        await self.run(lambda: self._apply_all(joints, lambda name: self._controller.set_torque(name, enabled)))

    async def save_zero(self, joint: str) -> None:
        """정지한 현재 자세를 관절 영점으로 저장한다."""
        self._require_idle_calibration()
        await self.run(lambda: self._update_calibration(lambda: self._calibration.save_zero(joint)))


class MoveRequest(BaseModel):
    """절대 이동 요청이다."""

    joint: str
    angle_deg: float
    speed_deg_s: float = 10.0


class JogRequest(BaseModel):
    """상대 이동 요청이다."""

    joint: str
    delta_deg: float
    speed_deg_s: float = 10.0


class ZeroMoveRequest(BaseModel):
    """모든 관절을 저장된 영점으로 이동시키는 요청이다."""

    speed_deg_s: float = 10.0


class TorqueRequest(BaseModel):
    """관절 하나 또는 전체의 토크 전환 요청이다."""

    joint: str | None = None
    enabled: bool


class JointRequest(BaseModel):
    """관절 하나 또는 전체를 대상으로 하는 요청이다."""

    joint: str | None = None


def create_app(port: str | None, baudrate: int | None) -> FastAPI:
    """포트를 열어 제어기를 연결한 뒤 화면과 명령 경로를 제공하는 앱을 만든다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """서버가 쓰는 통신 자원을 열고 진행 중인 작업을 마친 뒤 닫는다."""
        settings = resolve_port_settings(port, baudrate)
        bus = STS3215Bus(settings.port, settings.baudrate).open()
        try:
            app.state.console = Console(JointController(bus), MotorCalibration(bus))
            app.state.port = settings.port
            app.state.console.start()
            try:
                yield
            finally:
                await app.state.console.close()
        finally:
            bus.close()

    app = FastAPI(title="인치웜 관절 콘솔", lifespan=lifespan)

    def console() -> Console:
        """이 서버가 공유하는 관절 콘솔을 반환한다."""
        return app.state.console

    def targets(joint: str | None) -> tuple[str, ...]:
        """요청이 가리키는 관절을 정하며 생략하면 전체를 대상으로 한다."""
        if joint is None:
            return console().joint_names
        if joint not in console().joint_names:
            raise HTTPException(status_code=404, detail=f"알 수 없는 관절입니다: {joint}")
        return (joint,)

    @app.get("/")
    async def index() -> FileResponse:
        """조작 화면을 전달한다."""
        return FileResponse(INDEX_PATH)

    @app.get("/api/joints")
    async def joints() -> dict:
        """관절 순서와 연결한 포트를 알려 준다."""
        return {"joints": list(console().joint_names), "port": app.state.port}

    @app.post("/api/move")
    async def move(request: MoveRequest) -> dict:
        """선택한 관절을 목표 각도로 이동시킨다."""
        targets(request.joint)
        try:
            await console().move(request.joint, request.angle_deg, request.speed_deg_s)
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True, "status": "accepted"}

    @app.post("/api/jog")
    async def jog(request: JogRequest) -> dict:
        """선택한 관절을 현재 각도에서 상대 이동시킨다."""
        targets(request.joint)
        try:
            await console().jog(request.joint, request.delta_deg, request.speed_deg_s)
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True, "status": "accepted"}

    @app.post("/api/move-zero")
    async def move_zero(request: ZeroMoveRequest) -> dict:
        """영점 설정을 바꾸지 않고 전체 관절에 영점 이동을 명령한다."""
        try:
            await console().move_all_zero(request.speed_deg_s)
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True, "status": "accepted"}

    @app.post("/api/stop")
    async def stop(request: JointRequest) -> dict:
        """관절 하나 또는 전체를 현재 위치에서 멈춘다."""
        try:
            await console().stop(targets(request.joint))
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True}

    @app.post("/api/torque")
    async def torque(request: TorqueRequest) -> dict:
        """관절 하나 또는 전체의 토크를 켜거나 끈다."""
        try:
            await console().set_torque(targets(request.joint), request.enabled)
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True}

    @app.post("/api/zero")
    async def zero(request: JointRequest) -> dict:
        """선택한 관절의 현재 자세를 영점으로 저장한다."""
        if request.joint is None:
            raise HTTPException(status_code=400, detail="영점을 저장할 관절을 선택해 주세요.")
        targets(request.joint)
        try:
            await console().save_zero(request.joint)
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True}

    @app.post("/api/calibrate-zero")
    async def calibrate_zero() -> dict:
        """등록된 모든 모터의 현재 자세를 중점과 관절 영점으로 함께 설정한다."""
        try:
            names = await console().calibrate_all_zero()
        except BUS_ERRORS as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True, "joints": list(names)}

    @app.websocket("/ws")
    async def state(websocket: WebSocket) -> None:
        """공통 조회 결과를 전달하고 화면 연결이 끊기면 즉시 구독을 해제한다."""
        await websocket.accept()
        async with console().subscribe() as updates:
            async def send_updates() -> None:
                """이 화면에 가장 최근의 상태를 전달한다."""
                while True:
                    await websocket.send_json(await updates.get())

            async def receive_disconnect() -> None:
                """클라이언트의 연결 종료를 기다린다."""
                while True:
                    await websocket.receive_text()

            tasks = [asyncio.create_task(send_updates()), asyncio.create_task(receive_disconnect())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except WebSocketDisconnect:
                pass
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    return app


def main(argv: list[str] | None = None) -> int:
    """조작 화면을 여는 로컬 서버를 실행한다."""
    parser = argparse.ArgumentParser(description="인치웜 관절 콘솔 서버")
    parser.add_argument("--port", help="생략하면 저장된 직렬 포트 사용")
    parser.add_argument("--baudrate", type=int, help="생략하면 선택한 포트의 통신 속도 사용")
    parser.add_argument("--http-port", type=int, default=8000, help="화면을 여는 주소의 포트")
    args = parser.parse_args(argv)
    print(f"화면 주소: http://127.0.0.1:{args.http_port}", flush=True)
    uvicorn.run(create_app(args.port, args.baudrate), host="127.0.0.1", port=args.http_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
