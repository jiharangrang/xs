"""삽입 자료의 물리적 목표, 분할 독립성, 정답 균형과 오검출 집계를 검증한다."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ARTIFACTS
from insertion_data import X_INTERVALS, candidate_pool, select_training
from insertion_experiment import detailed_metrics
from scene import DistanceScene


class InsertionTests(unittest.TestCase):
    """학습 결과를 외우지 않고 자료 생성과 평가의 계약을 확인한다."""

    def test_splits_and_reachable_targets(self):
        """각 분할의 IK 결과가 서로 다른 구간에 있고 실제 FK가 목표와 일치한다."""
        scene = DistanceScene()
        families = []
        for index, split in enumerate(X_INTERVALS):
            pool, info = candidate_pool(split, 2, 8, 501 + index, ARTIFACTS / "scene/trace.npz", 1)
            self.assertEqual(info["ik_failed"], 0)
            self.assertEqual(len(pool["q"]), 16)
            self.assertEqual(set(pool["phase"]), {0, 1, 2, 3})
            families.append(set(pool["family"]))
            for q, position in zip(pool["q"], pool["target_position"], strict=True):
                scene.set_q(q)
                np.testing.assert_allclose(scene.data.site("tip_R").xpos, position, atol=1.1e-5)
                self.assertTrue(any(lower <= position[0] <= upper for lower, upper in X_INTERVALS[split]))
                self.assertTrue(np.isfinite(scene.distance(q)))
        self.assertFalse(families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
        for left, a in enumerate(X_INTERVALS.values()):
            for right, b in enumerate(X_INTERVALS.values()):
                if left >= right:
                    continue
                for lo_a, hi_a in a:
                    for lo_b, hi_b in b:
                        self.assertTrue(hi_a < lo_b or hi_b < lo_a)

    def test_teacher_strata_and_shortage(self):
        """정답 부호별 표본 수와 중복 방지를 검증하고 부족한 후보는 거부한다."""
        phase = np.repeat(np.arange(4), 300)
        distances = np.tile(np.repeat([.003, -.002, .02], 100), 4)
        pool = {"phase": phase, "d": distances, "identity": np.arange(len(phase))}
        selected = select_training(pool, 600, np.random.default_rng(1))
        self.assertEqual(len(np.unique(selected["identity"])), 600)
        self.assertEqual(np.bincount(selected["phase"]).tolist(), [60, 180, 180, 180])
        for stage in (1, 2, 3):
            d = selected["d"][selected["phase"] == stage]
            self.assertEqual(int(((d > 0) & (d <= .005)).sum()), 70)
            self.assertEqual(int(((d <= 0) & (d >= -.005)).sum()), 70)
        with self.assertRaisesRegex(ValueError, "후보 부족"):
            select_training(pool, 6000, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            select_training(pool, 90, np.random.default_rng(1))

    def test_false_collision_and_missed_collision_are_distinct(self):
        """안전 자세 과검출 감소가 충돌 누락 증가를 숨기지 않는지 확인한다."""
        result = detailed_metrics(np.array([-.001, .001, .003, -.003]),
                                  np.array([.001, -.001, .003, -.003]))
        self.assertEqual(result["false_collision_count"], 1)
        self.assertEqual(result["missed_collision_count"], 1)
        self.assertEqual(result["near_1mm"]["accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
