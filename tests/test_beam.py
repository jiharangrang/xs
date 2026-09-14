"""알려진 3차원 빔과 가림·잡음 조건으로 평면 및 모서리 추정을 검증한다."""

import unittest

import numpy as np

from perception.beam import BeamDetectionError, estimate_beam
from perception.depth import depth_points, project_points


PROFILE = {"width": 320, "height": 240,
           "intrinsics": {"fx": 300.0, "fy": 310.0, "cx": 159.0, "cy": 119.0}}


def synthetic_beam(angle_deg: float = 0, *, noisy: bool = False) -> tuple[np.ndarray, dict]:
    r"""알려진 평면과 폭을 가진 무한 직선 빔의 광학 깊이를 생성한다.

    $$z=\frac{n^Tp_0}{n^Tr},\quad |(zr-p_0)^Tb|\leq w/2$$

    n은 법선, p_0는 평면 위 점, b는 폭 방향, w는 실제 폭이다.
    """
    rows, cols = np.indices((240, 320))
    rays = np.stack(((cols - 159) / 300, (rows - 119) / 310, np.ones_like(cols)), axis=-1)
    normal = np.array([0.12, -0.18, -1.0])
    normal /= np.linalg.norm(normal)
    angle = np.deg2rad(angle_deg)
    axis = np.array([np.sin(angle), np.cos(angle), 0.0])
    # 길이 방향을 평면 위에 투영: $$t'=t_0-n(n^Tt_0)$$
    axis -= normal * (normal @ axis)
    # 길이 방향의 단위 벡터: $$t=t'/\|t'\|$$
    axis /= np.linalg.norm(axis)
    # 빔의 폭 방향: $$b=n\times t$$
    across = np.cross(normal, axis)
    point = np.array([0.01, -0.005, 0.3])
    # 광선과 평면 교점의 광학 깊이: $$z=(n^Tp_0)/(n^Tr)$$
    depth = (normal @ point) / (rays @ normal)
    # 교점의 카메라 좌표: $$p=zr$$
    points = rays * depth[..., None]
    # 폭 바깥의 점을 제거: $$|(p-p_0)^Tb|>w/2$$
    depth[np.abs((points - point) @ across) > 0.035] = np.nan
    if noisy:
        rng = np.random.default_rng(42)
        depth += rng.normal(0, 0.0003, depth.shape)
        depth[rng.random(depth.shape) < 0.03] = np.nan
        depth[rng.random(depth.shape) < 0.03] += 0.03
    # 중심선에서 카메라에 가장 가까운 정답 점: $$c=p_0-t(t^Tp_0)$$
    center = point - axis * (axis @ point)
    return depth.astype(np.float32), {"normal": normal, "axis": axis, "center": center,
                                      "offset": -normal @ point, "width": 0.07}


class BeamTests(unittest.TestCase):
    """실측값과 추정할 수 없는 항목을 구분하는지 확인한다."""

    def test_depth_pixel_round_trip_and_invalid_values(self):
        """깊이에서 복원한 점을 투영하면 원래 픽셀로 돌아오고 무효 값은 유지된다."""
        profile = {"width": 3, "height": 2,
                   "intrinsics": {"fx": 150, "fy": 170, "cx": 0.8, "cy": 0.6}}
        depth = np.array([[0.2, 0, np.nan], [0.3, 0.4, 0.5]])
        points = depth_points(depth, profile)
        valid = np.isfinite(points[..., 0])
        rows, columns = np.indices(depth.shape)
        np.testing.assert_allclose(project_points(points[valid], profile),
                                   np.stack((columns, rows), axis=-1)[valid], atol=1e-12)
        self.assertTrue(np.isnan(points[0, 1:]).all())

    def test_tilted_beams_with_noise_and_missing_depth(self):
        """다른 방향의 빔에서 잡음과 깊이 누락이 있어도 평면·폭·중심선을 복원한다."""
        for angle in (0, 35, 80):
            with self.subTest(angle=angle):
                depth, truth = synthetic_beam(angle, noisy=True)
                result = estimate_beam(depth, PROFILE)
                self.assertEqual(result.status, "two_edges")
                self.assertAlmostEqual(result.width_m, truth["width"], delta=0.0025)
                self.assertAlmostEqual(result.plane_offset_m, truth["offset"], delta=0.0007)
                self.assertGreater(abs(result.axis @ truth["axis"]), np.cos(np.deg2rad(1)))
                self.assertGreater(result.normal @ truth["normal"], np.cos(np.deg2rad(0.5)))
                self.assertLess(np.linalg.norm(result.centerline_point_m - truth["center"]), 0.002)
                self.assertAlmostEqual(float(result.normal @ result.axis), 0, delta=1e-8)
                self.assertAlmostEqual(float(result.normal @ result.centerline_point_m) + result.plane_offset_m,
                                       0, delta=1e-8)

    def test_image_and_roi_edges_are_not_beam_edges(self):
        """화면 전체의 평면과 잘라낸 ROI에서 허위 빔 폭을 만들지 않는다."""
        depth = np.full((240, 320), 0.25, dtype=np.float32)
        for roi in (None, (30, 25, 285, 210)):
            result = estimate_beam(depth, PROFILE, roi=roi)
            self.assertEqual(result.status, "plane_only")
            self.assertIsNone(result.width_m)
            self.assertIsNone(result.axis)

    def test_internal_hole_does_not_become_a_beam_boundary(self):
        """넓은 평면 안의 깊이 구멍을 빔 모서리로 해석하지 않는다."""
        depth = np.full((240, 320), 0.25, dtype=np.float32)
        depth[70:170, 100:220] = np.nan
        result = estimate_beam(depth, PROFILE)
        self.assertEqual(result.status, "plane_only")

    def test_single_edge_leaves_width_and_center_unknown(self):
        """한 모서리만 보일 때 관측하지 못한 폭과 중심선을 채우지 않는다."""
        depth = np.full((240, 320), np.nan, dtype=np.float32)
        depth[:, :180] = 0.25
        result = estimate_beam(depth, PROFILE)
        self.assertEqual(result.status, "axis_only")
        self.assertIsNone(result.width_m)
        self.assertIsNone(result.centerline_point_m)

    def test_square_patch_does_not_choose_arbitrary_length_axis(self):
        """길이와 폭을 구분하기 어려운 사각 영역은 방향이 모호하다고 표시한다."""
        depth = np.full((240, 320), np.nan, dtype=np.float32)
        depth[60:180, 100:220] = 0.25
        result = estimate_beam(depth, PROFILE)
        self.assertEqual(result.status, "ambiguous_edges")
        self.assertIsNone(result.width_m)

    def test_missing_depth_and_bad_inputs_fail_clearly(self):
        """유효 깊이가 없거나 입력 설정이 잘못되면 명확히 실패한다."""
        with self.assertRaises(BeamDetectionError):
            estimate_beam(np.full((240, 320), np.nan), PROFILE)
        depth, _ = synthetic_beam()
        for kwargs in ({"roi": (-1, 0, 200, 200)}, {"axis_hint": [0, 0, 0]},
                       {"plane_threshold_m": 0}, {"depth_range_m": (0.7, 0.1)}):
            with self.assertRaises(ValueError):
                estimate_beam(depth, PROFILE, **kwargs)


if __name__ == "__main__":
    unittest.main()
