"""모서리 관측을 주기적으로 갱신하며 목표 자세를 추종하는 단계의 실행을 공유한다.
관절 도착을 기다리지 않는 관측 주기·기준 모서리·도착 자세 유지 기능을 제공한다.
"""

import asyncio
import time

import numpy as np

from gui.observed_lift import ObservedLiftSession
from kinematics.joints import ARM_JOINT_NAMES
from perception.beam_edge_observation import EdgeReference


class ObservedBeamSession(ObservedLiftSession):
    """최종 상승과 삽입에서 같은 관측 주기와 새 영상 기반 완료 확인을 사용한다."""

    _bounded_execution = False
    _reference_stages = (4,)

    def __init__(self, console, settings, *, stage, label, observation_interval_s=.5):
        """단계 번호와 관측 주기를 받아 공통 실행 상태를 준비한다."""
        super().__init__(console, settings, stage=stage, label=label)
        if not np.isfinite(observation_interval_s) or observation_interval_s <= 0:
            raise ValueError("재관측 간격은 유한한 양수여야 합니다.")
        self.observation_interval_s = observation_interval_s

    def _reset_observation_cycle(self):
        """계획기를 준비한 뒤 선행 모서리 참고값과 목표 유지 상태를 초기화한다."""
        self._reference_world = self._previous_reference()
        self._next_observation_s = 0.
        self._holding_goal = False

    def _previous_reference(self):
        """선행 단계의 도착 관측에서 모서리 참고값을 얻으며 없으면 새 양쪽 모서리를 기다린다."""
        for stage in self._reference_stages:
            previous = getattr(self.console, f"stage{stage}").status()
            if previous["state"] != "REACHED" or not previous.get("arrival") or not previous.get("observation"):
                continue
            reference = self._reference_from_status(previous)
            if reference is not None:
                return reference
        return None

    def _reference_from_status(self, previous):
        r"""도착 관측과 당시 실측 관절각으로 월드 모서리를 복원한다.

        $$p_W=R_{WC}p_C+p_{WC}$$
        """
        try:
            positions = previous["arrival"]["positions_deg"]
            # 관측 당시의 몸통 각도를 기구학 입력으로 변환: $$q_{rad}=q_{deg}\pi/180$$
            q_rad = np.deg2rad([positions[name] for name in ARM_JOINT_NAMES])
            camera = self.planner.solver.fk.depth_camera_pose(q_rad)
            observed = previous["observation"]
            reference = EdgeReference(np.asarray(observed["edge_point_m"]), np.asarray(observed["axis"]),
                                      np.asarray(observed["outward"]), np.asarray(observed["normal"]),
                                      observed["plane_offset_m"], observed["width_m"])
            return reference.transformed(camera[:3, :3], camera[:3, 3])
        except (KeyError, ValueError, TypeError):
            return None

    async def _observation_states(self, deadline):
        """관절 도착 여부와 무관하게 다음 주기에 새 피드백과 영상을 확인한다."""
        while time.monotonic() < self._next_observation_s:
            await self.console.run(self._guard)
            await asyncio.sleep(min(.05, max(0., self._next_observation_s - time.monotonic())))
        states = await self._snapshot()
        self._next_observation_s = time.monotonic() + self.observation_interval_s
        return states

    def _observation_options(self, states):
        r"""현재 카메라 좌표의 모서리 참고값과 고정턱 방향을 준비한다.

        $$t_{CW}=-R_{WC}^Tp_{WC}$$
        """
        q_current = self._angles(states)
        grippers = self._grippers(states)
        camera = self.planner.solver.fk.depth_camera_pose(q_current)
        # 월드 기준 모서리를 현재 카메라로 옮길 이동량: $$t_{CW}=-R_{WC}^Tp_{WC}$$
        translation = -camera[:3, :3].T @ camera[:3, 3]
        reference = None if self._reference_world is None else self._reference_world.transformed(camera[:3, :3].T, translation)
        return {"outward_hint": self.planner.outward_hint(q_current, grippers), "reference": reference}

    def _feedback_unchanged(self, before, after):
        """이동 중에도 관측을 사용하며 관절 정지·도착 판정을 요구하지 않는다."""
        self._body(after)
        self._grippers(after)
        return True

    async def _confirm_height(self, arrival, states):
        """목표를 관측하면 진행 중 목표를 현재각으로 바꾼 후 새 영상으로 재확인한다."""
        if self._holding_goal:
            return True
        targets = {s["name"]: s["position_deg"] for s in self._body(states)}
        receipt = await self.console.move_pose(targets, remember=False, owner=self, guard=self._command_guard,
                                                speed_deg_s=1., acceleration_deg_s2=10.)
        self._owned_ids.update(receipt["command_ids"])
        self._expected_ids.update(self._owned_ids)
        self._holding_goal = True
        self._next_observation_s = time.monotonic() + self.observation_interval_s
        self._update("OBSERVING", "현재 자세를 유지하며 새 영상으로 목표 도착을 확인합니다.",
                     targets_deg=receipt["targets_deg"], step=self._status["step"] + 1,
                     motion_kind="hold", holding_goal=True)
        return False
