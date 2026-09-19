"""4단계 출발 횡위치의 복원·저장과 실제 짧은 삽입 사례의 남은 거리를 검증한다."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kinematics.fk import ForwardKinematics
from kinematics.joints import ARM_JOINT_NAMES
from planning.insertion_reference import (LateralReference, load_lateral_reference,
                                          reference_from_stage4, save_lateral_reference)


def recorded_stage4():
    """RGB 중앙에서 조기 완료한 실행의 실제 4단계 출발 위치와 도착 관측을 반환한다."""
    return {"state": "REACHED", "run_id": "2fd59272c68c4976b20f9dc3cd1d8434",
            "arrival": {"start_tip_world_m": [.1930143840318622, -.03417202643448342, .22063076756269415],
                        "positions_deg": dict(zip(ARM_JOINT_NAMES,
                            [-5.361328125, -50.2734375, 10.986328125, 70.576171875,
                             3.955078125, 68.115234375, 15.64453125], strict=True))},
            "observation": {"normal": [.0018599293576145905, -.00047878959191656077, -.9999981557099549],
                            "outward": [.964818365470927, .2629120335122207, .00166861753488454]}}


class InsertionReferenceTests(unittest.TestCase):
    """명령 누적량이나 RGB 중앙 대신 출발 위치와 빔 횡축만 복귀 기준으로 쓰는지 확인한다."""

    def test_recorded_rgb_centered_pose_still_requires_about_ten_mm(self):
        """실물에서 덜 삽입된 도착 자세를 재현하면 아직 남은 횡복귀 거리를 얻는다."""
        fk = ForwardKinematics()
        reference = reference_from_stage4(recorded_stage4(), fk)
        q = np.deg2rad([-14.23828125, -62.9296875, 3.779296875, 71.3671875,
                        -2.197265625, 56.25, 15.732421875])
        tip = fk.forward(q).T_world_tip_R[:3, 3]
        self.assertAlmostEqual(reference.remaining(tip), .0097, delta=.001)
        self.assertEqual(reference.source_run_id, recorded_stage4()["run_id"])

    def test_height_and_longitudinal_changes_do_not_reset_lateral_goal(self):
        """높이와 빔 길이 방향 이동은 복귀 거리에 더하지 않으며 통과 시 부호가 바뀐다."""
        reference = LateralReference((.2, -.03, .22), (0., 2., 0.), "run")
        self.assertAlmostEqual(reference.remaining((.3, -.006, .27)), .024)
        self.assertAlmostEqual(reference.remaining((.3, -.033, .27)), -.003)

    def test_persistence_keeps_same_goal_and_source(self):
        """저장과 재시작 후에도 위치·방향·원본 실행이 동일하게 복원된다."""
        reference = LateralReference((.2, -.03, .22), (0., 1., 0.), "run")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            save_lateral_reference(reference, path)
            self.assertEqual(load_lateral_reference(path), reference)
            self.assertFalse(path.with_suffix(".tmp").exists())
            document = json.loads(path.read_text())
            document["kind"] = "rgb_center"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_lateral_reference(path)

    def test_incomplete_or_invalid_reference_has_no_fallback(self):
        """누락·중단된 실행이나 유효하지 않은 좌표를 임의의 기본 목표로 바꾸지 않는다."""
        fk = ForwardKinematics()
        for state in ("IDLE", "FAILED", "STOPPED", "OBSERVING"):
            with self.subTest(state=state), self.assertRaises(ValueError):
                reference_from_stage4({**recorded_stage4(), "state": state}, fk)
        for point, outward in (([0., 0.], [0., 1., 0.]),
                               ([0., 0., float("nan")], [0., 1., 0.]),
                               ([0., 0., 0.], [0., 0., 0.])):
            with self.assertRaises(ValueError):
                LateralReference(point, outward, "run")


if __name__ == "__main__":
    unittest.main()
