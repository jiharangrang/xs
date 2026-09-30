"""단계별 명령 속도와 동기 관절 전송·마지막 목표 도착 대기를 공유해요."""

import asyncio
import time

from hardware.sts3215 import MotorError


GRIPPER_SPEED_DEG_S = 30.
STAGE1_4_BODY_SPEED_DEG_S = 20.


class JointCommandMixin:
    """관측 세션의 피드백·소유권 검사와 공통 모터 제어기를 연결한다."""

    def _send_guard(self):
        """직렬 전송 직전에 취소·소유권과 지지·개방 상태를 확인한다."""
        self._guard()
        self._body([self.console._read_joint(name) for name in self.console.joint_names])
        self._guard()

    async def _send(self, targets, message, *, phase, speed_deg_s):
        """공통 동기 전송을 사용하고 이번 실행의 모든 명령 소유권을 유지한다."""
        receipt = await self.console.move_pose(targets, remember=False, owner=self, guard=self._send_guard,
                                               speed_deg_s=speed_deg_s, acceleration_deg_s2=20.)
        self._owned_ids.update(receipt["command_ids"])
        self._expected_ids.update(receipt["command_ids"])
        self._status["targets_deg"].update(receipt["targets_deg"])
        self._update("MOVING", message, phase=phase, step=self._status["step"] + 1,
                     targets_deg=dict(self._status["targets_deg"]))
        return receipt["targets_deg"]

    async def _wait_targets(self, targets):
        """보낸 목표의 실제 각도와 정지 상태를 기다린다."""
        deadline = time.monotonic() + self.settings.motion_timeout_s
        stable_since = None
        while time.monotonic() < deadline:
            states = await self._snapshot()
            by_name = {s["name"]: s for s in states}
            near = all(abs(by_name[name]["position_deg"] - angle) <= self.settings.tracking_tolerance_deg
                       and by_name[name]["speed_deg_s"] == 0 and by_name[name]["torque"] is True
                       for name, angle in targets.items())
            if near:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= self.settings.settle_s:
                    return states
            else:
                stable_since = None
            await asyncio.sleep(.1)
        raise MotorError("이동 목표에 도착하지 못했습니다. 현재 위치를 확인해 주세요.")
