"""저장된 관측 시작 자세와 1단계 추적기를 기존 서버의 모터 통신·상태 스트림에 연결한다.
카메라 연결과 이후 단계의 실행은 담당하지 않는다.
"""

import asyncio
import json
from pathlib import Path

from fastapi import HTTPException
import numpy as np

from control.stage1 import Stage1Tracker
from gui.joint_command import STAGE1_4_BODY_SPEED_DEG_S
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES, as_joint_angles


DEFAULT_POSE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "stage1_observation_pose.json"


class Stage1Session:
    """명령은 기존 콘솔로 전송하고 조회 결과로만 1단계 완료를 갱신한다."""

    def __init__(self, console, *, pose_path: Path = DEFAULT_POSE_PATH) -> None:
        """장치를 열지 않고 저장 자세와 콘솔 및 독립 추적기를 연결한다."""
        self.console = console
        self.pose_path = pose_path
        self.tracker = Stage1Tracker()
        self._lock = asyncio.Lock()
        self._log_path = None
        self._stop_error = None
        self._start_task = None

    def status(self) -> dict:
        """현재 단계와 공통 모터 로그의 위치를 전달한다."""
        return {**self.tracker.status(), "pose_path": str(self.pose_path),
                "log_path": self._log_path, "stop_error": self._stop_error}

    def _targets(self) -> dict[str, float]:
        r"""저장된 1단계 관절각을 검증하고 몸통 관절만 도 단위로 반환한다.

        $$q_{deg}=q_{rad}180/\pi$$
        """
        document = json.loads(self.pose_path.read_text(encoding="utf-8"))
        if (not isinstance(document, dict) or document.get("format") != "xs.observation_pose.v1"
                or document.get("stage") != 1 or document.get("fixed_tip") != "tip_L"):
            raise ValueError("L 고정으로 만든 1단계 관측 시작 자세 파일이 필요합니다.")
        angles = as_joint_angles(document.get("q_rad"))
        # 저장된 라디안 목표를 공통 제어기의 도 단위로 변환: $$q_{deg}=q_{rad}180/\pi$$
        degrees = np.rad2deg(angles)
        return dict(zip(ARM_JOINT_NAMES, degrees.tolist(), strict=True))

    async def start(self) -> dict:
        """요청 연결이 끊겨도 전송한 명령의 도착 추적이 등록될 때까지 시작 작업을 유지한다."""
        if self._start_task is not None and not self._start_task.done():
            raise MotorError("1단계 시작 명령을 처리 중입니다.")
        self._start_task = asyncio.create_task(self._start())
        self._start_task.add_done_callback(self._finish_start)
        return await asyncio.shield(self._start_task)

    @staticmethod
    def _finish_start(task) -> None:
        """요청자가 연결을 끊은 시작 작업의 예외도 회수한다."""
        if not task.cancelled():
            task.exception()

    async def _start(self) -> dict:
        """확정된 자세를 한 번 전송하고 이번 명령의 도착 추적을 시작한다."""
        async with self._lock:
            if self.tracker.active:
                raise MotorError("이미 1단계 자세로 이동 중입니다.")
            targets = self._targets()
            receipt = await self.console.move_pose(targets, remember=False, speed_deg_s=STAGE1_4_BODY_SPEED_DEG_S)
            self.tracker.begin(receipt["command_ids"], receipt["targets_deg"])
            self._log_path = receipt["log_path"]
            self._stop_error = None
            return self.status()

    async def observe(self, states: list[dict]) -> None:
        """공통 조회 결과를 추적기에 전달하고 실패 시 아직 소유한 명령만 정지한다."""
        async with self._lock:
            if not self.tracker.active:
                return

            def update() -> None:
                """관절별 조회 사이에 들어온 명령까지 확인한 뒤 같은 통신 작업에서 완료를 판정한다."""
                for name, identifier in self.tracker.command_ids.items():
                    if self.console._controller.command_id(name) != identifier:
                        self.tracker.cancel(f"{name}: 조회 이후 다른 명령으로 1단계가 중단됐습니다.")
                        return
                self.tracker.update(states)

            await self.console.run(update)
            if self.tracker.state in ("FAILED", "STOPPED"):
                await self._stop_owned(dict(self.tracker.command_ids))

    async def _stop_owned(self, command_ids: dict[str, int]) -> None:
        """다른 조작으로 바뀐 명령과 그리퍼는 건드리지 않고 남은 1단계 명령을 정지한다."""
        def stop() -> None:
            """명령 소유권 확인과 정지를 같은 직렬 통신 작업 안에서 수행한다."""
            controller = self.console._controller
            owned = tuple(name for name, identifier in command_ids.items()
                          if controller.command_id(name) == identifier)
            self.console._apply_all(owned, controller.stop)

        try:
            await self.console.run(stop)
        except (ValueError, MotorError, OSError) as error:
            self._stop_error = str(error)

    async def stop(self) -> dict:
        """진행 중인 1단계를 중지하고 실제 정지 요청 결과를 반환한다."""
        if self._start_task is not None and not self._start_task.done():
            await asyncio.gather(asyncio.shield(self._start_task), return_exceptions=True)
        async with self._lock:
            if self.tracker.active:
                command_ids = dict(self.tracker.command_ids)
                self.tracker.cancel()
                await self._stop_owned(command_ids)
            return self.status()


def install_routes(app, console) -> None:
    """기존 자세 명령과 별도로 1단계 시작·조회·정지 경로를 등록한다."""
    @app.get("/api/stage1")
    async def status():
        """1단계 상태와 해당 실행의 완료 신호를 조회한다."""
        return console().stage1.status()

    @app.post("/api/stage1/start")
    async def start():
        """저장된 1단계 자세를 기존 모터 제어 통로로 실행한다."""
        try:
            return await console().stage1.start()
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage1/stop")
    async def stop():
        """1단계가 아직 소유하는 몸통 관절만 중지한다."""
        return await console().stage1.stop()
