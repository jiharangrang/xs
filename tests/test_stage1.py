"""1단계의 명령 식별·실측 도착·단일 완료 신호와 중단 상태를 검증한다."""

import unittest

from control.stage1 import Stage1Tracker
from kinematics.joints import ARM_JOINT_NAMES


class Stage1TrackerTests(unittest.TestCase):
    """통신 없이 공통 제어기가 판정한 최신 도착 상태를 단계 완료로 연결한다."""

    def setUp(self) -> None:
        """동일 실행의 명령과 신선한 도착 피드백을 준비한다."""
        self.now = 10.0
        self.tracker = Stage1Tracker(clock=lambda: self.now)
        self.ids = dict.fromkeys(ARM_JOINT_NAMES, 2)
        self.targets = dict.fromkeys(ARM_JOINT_NAMES, 5.0)
        self.tracker.begin(self.ids, self.targets)
        self.states = [{"name": name, "command_id": 2, "motion_status": "arrived",
                        "target_deg": 5.0, "position_deg": 5.0, "arrived_now": True,
                        "sampled_at_s": 10.1, "torque": True, "error": None}
                       for name in ARM_JOINT_NAMES]
        self.now = 10.2

    def test_arrival_signal_is_emitted_once_and_contains_observation_barrier(self) -> None:
        """도착 전에는 신호가 없고 도착 이후에는 실행 번호와 완료 확인 시각이 남는다."""
        self.assertIsNone(self.tracker.status()["arrival"])
        event = self.tracker.update(self.states)
        self.assertEqual(self.tracker.state, "ARRIVED")
        self.assertEqual(event.observed_at_s, self.now)
        self.assertEqual(event.command_ids, self.ids)
        self.assertEqual(event.positions_deg, self.targets)
        self.assertIsNone(self.tracker.update(self.states))
        snapshot = self.tracker.status()
        snapshot["arrival"]["command_ids"]["J1"] = 99
        self.assertEqual(self.tracker.status()["arrival"]["command_ids"]["J1"], 2)

    def test_old_commands_and_pre_start_feedback_cannot_complete(self) -> None:
        """이전 실행의 도착과 이번 실행 시작 전 표본을 건너뛴다."""
        for changes in ({"command_id": 1}, {"command_id": None}, {"sampled_at_s": 9.9}):
            with self.subTest(changes=changes):
                states = [{**state, **changes} for state in self.states]
                self.assertIsNone(self.tracker.update(states))
                self.assertEqual(self.tracker.state, "MOVING")

    def test_latched_arrival_does_not_replace_current_arrival(self) -> None:
        """과거 도착 표시는 남아 있어도 지금 허용 범위를 벗어난 관절을 기다린다."""
        self.states[0]["arrived_now"] = False
        self.assertIsNone(self.tracker.update(self.states))
        self.assertEqual(self.tracker.state, "MOVING")
        self.states[0]["arrived_now"] = True
        self.assertIsNotNone(self.tracker.update(self.states))

    def test_replacement_torque_off_and_changed_target_are_not_success(self) -> None:
        """새 명령·토크 해제·목표 변경은 이번 실행의 완료 신호를 만들지 않는다."""
        for changes in ({"command_id": 3}, {"torque": False}, {"target_deg": 6.0}, {"motion_status": "idle"}):
            with self.subTest(changes=changes):
                tracker = Stage1Tracker(clock=lambda: 10.)
                tracker.begin(self.ids, self.targets)
                states = [dict(state) for state in self.states]
                states[0].update(changes)
                self.assertIsNone(tracker.update(states))
                self.assertEqual(tracker.state, "STOPPED")

    def test_missing_feedback_and_errors_fail_without_arrival(self) -> None:
        """누락·통신 오류를 정상 완료와 구분한다."""
        for states in (self.states[:-1], [{**state, "error": "통신 오류"} for state in self.states]):
            with self.subTest(states=states):
                tracker = Stage1Tracker(clock=lambda: 10.)
                tracker.begin(self.ids, self.targets)
                self.assertIsNone(tracker.update(states))
                self.assertEqual(tracker.state, "FAILED")

    def test_joint_timeout_keeps_waiting_for_current_arrival(self) -> None:
        """개별 예상시간이 지나도 전체 기한 안에서는 최신 도착을 기다린다."""
        self.states[1].update(motion_status="timeout", arrived_now=False)
        self.states[3].update(motion_status="moving", arrived_now=False)
        self.assertIsNone(self.tracker.update(self.states))
        self.assertEqual(self.tracker.state, "MOVING")
        self.states[1]["arrived_now"] = True
        self.states[3]["arrived_now"] = True
        self.assertIsNotNone(self.tracker.update(self.states))
        self.assertEqual(self.tracker.state, "ARRIVED")

    def test_wait_timeout_and_duplicate_start(self) -> None:
        """중복 시작은 거부하고 전체 대기 시간이 지나면 도착 신호 없이 종료한다."""
        with self.assertRaisesRegex(ValueError, "이동 중"):
            self.tracker.begin(self.ids, self.targets)
        self.now = 56.0
        self.assertIsNone(self.tracker.update(self.states))
        self.assertEqual(self.tracker.state, "FAILED")


if __name__ == "__main__":
    unittest.main()
