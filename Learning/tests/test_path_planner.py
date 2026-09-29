"""충돌 제약이 IK 내부에 적용되고 두 거리 방식의 미분·보간 조건이 같은지 검사한다."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path_planner import CollisionAwareIK, PathSettings, dense_path


class LinearDistance:
    """거리 제약의 미분을 해석적으로 확인할 수 있는 선형 시험 함수다."""

    def __call__(self, q):
        r"""첫 두 관절의 선형 결합을 거리로 반환한다.

        $$d(q)=0.02+0.01q_1+0.005q_2$$
        """
        # 알려진 거리 기울기를 갖는 선형 함수: $$d=0.02+0.01q_1+0.005q_2$$
        return .02 + .01 * q[:, 0] + .005 * q[:, 1]


class PathPlannerTests(unittest.TestCase):
    """물리 모델의 새 학습 없이 경로 생성 어댑터의 핵심 계약을 확인한다."""

    def test_interpolation_keeps_endpoints_without_duplicate_junctions(self):
        """최종 재검사에 시작·끝과 각 보간 구간이 빠짐없이 들어가는지 확인한다."""
        q = np.array([[0., 0.], [1., 0.], [1., 1.]])
        actual = dense_path(q, 4)
        self.assertEqual(actual.shape, (9, 2))
        np.testing.assert_array_equal(actual[[0, 4, 8]], q)
        self.assertEqual(int(np.all(actual == [1., 0.], axis=1).sum()), 1)
        np.testing.assert_allclose(actual[1], [.25, 0.])

    def test_edge_jacobian_matches_analytic_distance(self):
        """보간 비율과 관절 제한을 포함한 차분 결과를 알려진 기울기와 비교한다."""
        solver = CollisionAwareIK(LinearDistance())
        start = np.zeros(7)
        endpoint = np.full(7, .01)
        expected = np.zeros((4, 7))
        expected[:, 0] = [.025, .05, .075, .1]
        expected[:, 1] = [.0125, .025, .0375, .05]
        np.testing.assert_allclose(solver.edge_jacobian(endpoint, start), expected, atol=1e-10)
        endpoint[0] = solver._limits[0, 1]
        np.testing.assert_allclose(solver.edge_jacobian(endpoint, start), expected, atol=1e-10)

    def test_pose_match_does_not_bypass_collision_constraint(self):
        """목표 자세가 이미 일치해도 거리 조건에 실패하면 IK를 거부한다."""
        solver = CollisionAwareIK(lambda q: np.full(len(q), -.001))
        q = np.zeros(7)
        target = solver.fk.forward(q).T_tip_L_tip_R
        result = solver.solve(target, q)
        self.assertFalse(result.success)
        self.assertIn("거리 조건", result.message)
        with self.assertRaises(RuntimeError):
            solver.guard_segment(np.stack((q, q)))

    def test_validation_density_cannot_be_coarser(self):
        """최종 메시 재검사를 계획 중 검사보다 성기게 만드는 설정을 거부한다."""
        with self.assertRaises(ValueError):
            PathSettings(edge_subdivisions=4, validation_subdivisions=2)


if __name__ == "__main__":
    unittest.main()
