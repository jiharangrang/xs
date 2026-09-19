"""앞 그리퍼를 유지한 채 뒷 그리퍼를 열고 옆으로 빼서 당긴 뒤 빔 쪽으로 복귀한다.
현재각 유지와 개방 도착을 먼저 확인하며 어떤 종료 경로에서도 재파지하지 않는다.
"""

import asyncio
import time

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from gui.continuous_motion import ContinuousJointMotion
from gui.joint_command import GRIPPER_SPEED_DEG_S, JointCommandMixin
from gui.observed_motion import MotionStopped, ObservedMotionSession, ObservedMotionSettings
from hardware.camera import CameraError
from hardware.sts3215 import MotorError
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam import BeamDetectionError
from perception.beam_edge_observation import observe_beam_edge
from planning.motion_path import MotionPath, save_path
from planning.rear_pull import DEFAULT_PULL_MM, REAR_OPEN_DEG, REAR_SIDE_YAW_DEG, RearPullPlanner


class RearPullRequest(BaseModel):
    """한 번 당길 거리를 밀리미터로 받는다."""

    distance_mm: float = Field(default=DEFAULT_PULL_MM, gt=0, le=100, allow_inf_nan=False)


class Stage7Session(JointCommandMixin, ObservedMotionSession):
    """오른쪽 지지를 확인하고 개방과 몸통 이동을 순서대로 실행한다."""

    def __init__(self, console, *, planner=None, observer=observe_beam_edge, settings=None, continuous_motion=None):
        """실행 중 장치에 접근할 계획기와 관측기를 준비한다."""
        super().__init__(console, settings or ObservedMotionSettings(timeout_s=180., motion_timeout_s=25.),
                         stage=7, label="뒷그리퍼 당기기")
        self.planner, self.observer = planner, observer
        self.continuous_motion = continuous_motion if continuous_motion is not None else ContinuousJointMotion()
        self._holding_body = False
        self._rear_open = False
        self._anchor = None
        self._return_reference = None
        self._status.update(distance_mm=DEFAULT_PULL_MM, planned_progress_mm=0., path=None,
                            fixed_tip="tip_R", fixed_pose=None, phase="waiting", rear_open_deg=REAR_OPEN_DEG,
                            side_exit_deg=REAR_SIDE_YAW_DEG, side_exit_completed=False,
                            side_return_completed=False, return_reference=None,
                            side_return_distance_mm=None, planned_return_mm=0.,
                            pull_execution="continuous", pull_period_s=self.continuous_motion.settings.period_s)

    async def start(self, distance_mm=DEFAULT_PULL_MM):
        """입력 거리와 중복 실행을 확인한 뒤 개방·당김·횡복귀 작업을 시작한다."""
        request = RearPullRequest(distance_mm=distance_mm)
        if self.active:
            raise MotorError("이미 뒷그리퍼 당기기 중입니다.")
        await super().start()
        self._holding_body = False
        self._rear_open = False
        self._anchor = None
        self._return_reference = None
        self._status.update(distance_mm=request.distance_mm, planned_progress_mm=0., path=None,
                            fixed_pose=None, phase="planning", side_exit_completed=False,
                            side_return_completed=False, return_reference=None,
                            side_return_distance_mm=None, planned_return_mm=0.)
        return self.status()

    def _body(self, states):
        """앞 그리퍼의 지지와 현재 몸통 피드백을 확인한다."""
        by_name = {s["name"]: s for s in states}
        for name in (*ARM_JOINT_NAMES, "G_L", "G_R"):
            state = by_name.get(name)
            if (state is None or state.get("error") or state.get("position_deg") is None
                    or state.get("speed_deg_s") is None
                    or not np.isfinite(state["position_deg"]) or not np.isfinite(state["speed_deg_s"])):
                raise MotorError(f"{name}: 관절 피드백을 확인해 주세요.")
            if (name == "G_R" or self._holding_body and name in ARM_JOINT_NAMES) and state.get("torque") is not True:
                raise MotorError(f"{name}: 지지를 유지할 토크가 꺼져 있습니다.")
        front = by_name["G_R"]
        if not 0. <= front["position_deg"] <= 10.:
            raise MotorError("앞 그리퍼를 먼저 잠그고 빔에 고정됐는지 확인해 주세요.")
        if self._rear_open:
            rear = by_name["G_L"]
            if rear.get("torque") is not True or abs(rear["position_deg"] - REAR_OPEN_DEG) > 1.:
                raise MotorError("뒷그리퍼 개방 상태를 유지하지 못해 당김을 중지합니다.")
        return [by_name[name] for name in ARM_JOINT_NAMES]

    async def _plan_current(self, reader, filename, *, include_side_exit=True):
        """새 빔 방향과 현재 관절각으로 전체 경로를 검사하고 저장한다."""
        while True:
            before = await self._snapshot()
            barrier = time.monotonic()
            self._update("OBSERVING", "현재 자세와 새 빔 방향으로 당김 경로를 계산합니다.", phase="planning")
            try:
                observed = await asyncio.to_thread(self.observer, reader, barrier, outward_hint=np.array([1., 0., 0.]),
                                                  cancelled=lambda: self._cancelled)
            except (BeamDetectionError, CameraError) as error:
                await self.console.run(self._guard)
                self._update("OBSERVING", f"움직이지 않고 빔 방향을 다시 확인합니다 · {error}")
                await asyncio.sleep(.2)
                continue
            after = await self._snapshot()
            if (observed.first_frame_s <= barrier or time.monotonic() - observed.last_frame_s > .75
                    or np.max(np.abs(self._angles(after) - self._angles(before))) > np.deg2rad(.3)):
                continue
            plan = await asyncio.to_thread(self.planner.plan, self._angles(after), self._grippers(after),
                                           observed.axis, self._status["distance_mm"],
                                           include_side_exit=include_side_exit, anchor=self._anchor,
                                           return_reference=self._return_reference)
            fresh = await self._snapshot()
            if np.max(np.abs(self._angles(fresh) - self._angles(after))) > np.deg2rad(.3):
                continue
            path = self._logger.path.parent / self._status["run_id"] / filename
            await asyncio.to_thread(save_path, plan.motion, path)
            self._anchor = plan.pull.anchor
            self._return_reference = plan.return_reference
            self._status.update(observation=observed.as_dict(), path=str(path),
                                return_reference=plan.return_reference.as_dict(),
                                direction_world=plan.direction_world.tolist(),
                                fixed_pose=plan.pull.anchor.T_world_fixed_tip.tolist())
            return plan, fresh

    async def _pull_continuously(self, plan):
        """연속 목표를 전송하고 당김 전체의 끝에서만 실제 관절 도착을 확인한다."""
        return await self._move_continuously(plan.pull, phase="pulling", arrival_phase="pull_arrival",
                                            message="뒷그리퍼를 연 채 빔 방향으로 연속 당깁니다.",
                                            progress_key="planned_progress_mm", distance_mm=self._status["distance_mm"])

    async def _move_continuously(self, segment, *, phase, arrival_phase, message, progress_key, distance_mm):
        """당김과 횡복귀의 연속 전송을 공유하고 각 구간 끝에서만 도착을 확인한다."""
        async def send(q_rad):
            r"""현재 경로의 모든 몸통 목표를 한 동기 패킷으로 보낸다.

            $$q_{deg}=q_{rad}180/\pi$$
            """
            # 공통 모터 제어기로 보낼 도 단위 관절 목표: $$q_{deg}=q_{rad}180/\pi$$
            angles = dict(zip(ARM_JOINT_NAMES, np.rad2deg(q_rad).tolist(), strict=True))
            return await self._send(angles, message, phase=phase, speed_deg_s=3.)

        def progress(fraction):
            r"""실측 이동량과 구분해 명령한 경로의 진행량을 표시한다.

            $$s=d_{mm}u$$
            """
            # 명령한 경로 비율을 밀리미터로 표시: $$s=d_{mm}u$$
            self._status[progress_key] = distance_mm * fraction

        targets = await self.continuous_motion.run(segment, send, progress)
        self._update("MOVING", "이동 구간의 마지막 목표에 도착하는지 확인합니다.", phase=arrival_phase)
        return await self._wait_targets(targets)

    async def _return_laterally(self):
        r"""당김 후 새 실측각에서 복귀 경로를 갱신하고 열린 채 출발 횡위치로 이동한다.

        $$s_y=1000|u_W^T(p_L-p_0)|$$
        """
        self._update("OBSERVING", "당김 도착 자세에서 빔 쪽 횡복귀를 계산합니다.", phase="return_planning")
        while True:
            current = await self._snapshot()
            q_current = self._angles(current)
            segment = await asyncio.to_thread(self.planner.plan_side_return, q_current, self._grippers(current),
                                               self._anchor, self._return_reference)
            fresh = await self._snapshot()
            if np.max(np.abs(self._angles(fresh) - q_current)) <= np.deg2rad(.3):
                break
        path = self._logger.path.parent / self._status["run_id"] / "side_return.json"
        await asyncio.to_thread(save_path, MotionPath((segment,)), path)
        point = self._anchor.place(self.planner.solver.fk.forward(q_current)).T_world_tip_L[:3, 3]
        reference = self._return_reference
        # 명령할 횡복귀 거리의 밀리미터 변환: $$s_y=1000|u_W^T(p_L-p_0)|$$
        distance_mm = 1000 * abs(float(reference.outward_world @ (point - reference.point_world_m)))
        self._status.update(path=str(path), side_return_distance_mm=distance_mm)
        current = await self._move_continuously(segment, phase="side_return", arrival_phase="side_return_arrival",
                                                message="뒷그리퍼를 연 채 옆 빼기 전의 횡위치로 다시 넣습니다.",
                                                progress_key="planned_return_mm", distance_mm=distance_mm)
        self._status["side_return_completed"] = True
        return current

    async def _run(self):
        r"""현재각 유지·개방·외측 회전·연속 당김·횡복귀를 실행하고 열린 채 종료한다.

        $$q_{deg}=q_{rad}180/\pi$$
        """
        reader = None
        try:
            self._deadline = time.monotonic() + self.settings.timeout_s
            initial = await self._snapshot()
            self._expected_ids = {s["name"]: s.get("command_id") for s in initial}
            self._status["targets_deg"] = {s["name"]: s["position_deg"] for s in initial}
            if self.planner is None:
                self.planner = await asyncio.to_thread(RearPullPlanner)
            reader = self.console.camera.stream.reader()
            await asyncio.to_thread(reader.__enter__)
            _, current = await self._plan_current(reader, "before_open.json")
            hold = {s["name"]: s["position_deg"] for s in self._body(current)}
            targets = await self._send(hold, "현재 실측각으로 몸통 토크를 켜고 자세를 유지합니다.",
                                       phase="holding", speed_deg_s=3.)
            self._holding_body = True
            await self._wait_targets(targets)
            targets = await self._send({"G_L": REAR_OPEN_DEG}, "뒷그리퍼를 -120°로 엽니다.",
                                       phase="opening", speed_deg_s=GRIPPER_SPEED_DEG_S)
            await self._wait_targets(targets)
            self._rear_open = True
            plan, current = await self._plan_current(reader, "pull_open.json")
            # 외측 회전은 두 관절에만 같은 동기 패킷으로 전송: $$q_{deg}=q_{rad}180/\pi$$
            side_targets = dict(zip(("J1", "J7"), np.rad2deg(plan.side_exit.q_rad[-1, [0, 6]]).tolist(), strict=True))
            targets = await self._send(side_targets, f"J7 -{REAR_SIDE_YAW_DEG:g}° · J1 +{REAR_SIDE_YAW_DEG:g}°로 뒷그리퍼를 옆으로 뺍니다.",
                                       phase="side_exit", speed_deg_s=3.)
            await self._wait_targets(targets)
            self._status["side_exit_completed"] = True
            plan, current = await self._plan_current(reader, "pull_after_exit.json", include_side_exit=False)
            await self._pull_continuously(plan)
            current = await self._return_laterally()
            self._update("REACHED", "당김·횡복귀 완료 · 뒷그리퍼는 열린 상태입니다. 실제 삽입 위치를 확인해 주세요.",
                         phase="open_done", arrival={"confirmation": "joint_path_only", "closed": False,
                         "positions_deg": {s["name"]: s["position_deg"] for s in current}})
        except MotionStopped as error:
            self._update("STOPPED", str(error))
        except Exception as error:
            self._update("FAILED", str(error))
        finally:
            if self._status["state"] != "REACHED":
                await self._stop_owned()
            if reader is not None:
                try:
                    await asyncio.to_thread(reader.__exit__, None, None, None)
                except CameraError as error:
                    self._status["camera_error"] = str(error)


def install_routes(app, console):
    """뒷그리퍼 개방·당김·횡복귀의 시작·중지·상태 API를 등록한다."""
    @app.get("/api/stage7")
    async def status():
        """당김 진행 상태와 저장 경로를 반환한다."""
        return console().stage7.status()

    @app.post("/api/stage7/start")
    async def start(request: RearPullRequest):
        """현재 자세에서 뒷그리퍼를 열고 지정 거리만 당긴 뒤 빔 쪽으로 복귀한다."""
        try:
            return await console().stage7.start(request.distance_mm)
        except (ValueError, MotorError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/stage7/stop")
    async def stop():
        """다시 잠그지 않고 당김 동작을 중지한다."""
        return await console().stage7.stop()
