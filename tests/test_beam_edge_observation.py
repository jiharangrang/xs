"""깊이 영상의 부분 시야에서 같은 모서리를 추적하고 잘못된 경계를 거부하는지 검증한다."""

import time
import unittest

import numpy as np

from hardware.camera import CameraError, RGBDFrame
from perception.beam import BeamDetectionError, estimate_beam
from perception.beam_edge_observation import observe_beam_edge, select_edge
from perception.beam_observation import ObservationSettings
from test_beam import PROFILE
from test_beam_observation import FrameReader


def beam_depth(center, depth=.25):
    r"""카메라를 옆으로 옮긴 것과 같은 유한 폭의 평행 빔 깊이를 만든다.

    $$x=(u-c_x)z/f_x,\quad |x-c|\leq w/2$$
    """
    lens = PROFILE["intrinsics"]
    # 광학 깊이에서 영상 열의 횡좌표: $$x=(u-c_x)z/f_x$$
    x = (np.arange(PROFILE["width"]) - lens["cx"]) * depth / lens["fx"]
    # 빔 표면에 해당하는 픽셀의 깊이: $$z(u,v)=z_0\ \text{if}\ |x-c|\leq 0.035$$
    row = np.where(np.abs(x - center) <= .035, depth, np.nan)
    return np.tile(row, (PROFILE["height"], 1)).astype(np.float32)


class BeamEdgeObservationTests(unittest.TestCase):
    """양 모서리로 시작한 추적이 한 모서리만 보여도 연속성을 유지하는지 확인한다."""

    def observe(self, depth, *, reference=None, hint=None):
        """같은 새 깊이 세 장을 실제 검출·묶음 처리에 전달한다."""
        now = time.monotonic()
        frames = [RGBDFrame(np.zeros((2, 2, 3), np.uint8), depth, i, i, now + .01 * i) for i in (1, 2, 3)]
        return observe_beam_edge(FrameReader(frames), now, reference=reference,
                                 outward_hint=np.array([1., 0., 0.]) if hint is None else hint,
                                 clock=lambda: now + .1)

    def test_partial_view_follows_same_edge_with_known_beam_side(self):
        """반대 모서리가 시야 밖으로 나가도 같은 모서리로 위치를 갱신한다."""
        initial = self.observe(beam_depth(.01))
        self.assertFalse(initial.single_edge)
        reference = initial.reference().transformed(np.eye(3), np.array([-.12, 0., 0.]))
        partial = self.observe(beam_depth(-.11), reference=reference)
        self.assertTrue(partial.single_edge)
        self.assertAlmostEqual(partial.edge_point_m[0], -.075, delta=.0015)
        self.assertGreater(partial.outward[0], .999)
        self.assertAlmostEqual(partial.width_m, initial.width_m)

    def test_first_frame_cannot_guess_width_from_one_edge(self):
        """최초부터 한 모서리만 보이면 추정 폭으로 새 이동을 시작하지 않는다."""
        with self.assertRaises(CameraError):
            self.observe(beam_depth(-.11))

    def test_opposite_edge_and_other_plane_are_rejected(self):
        """반대편 모서리와 다른 높이의 평면을 같은 추적 대상으로 오인하지 않는다."""
        initial = self.observe(beam_depth(.01))
        reference = initial.reference()
        wrong_edge = estimate_beam(beam_depth(.16), PROFILE)
        with self.assertRaises(BeamDetectionError):
            select_edge(wrong_edge, PROFILE, outward_hint=initial.outward, reference=reference,
                        settings=ObservationSettings())
        with self.assertRaises(CameraError):
            self.observe(beam_depth(.01, depth=.28), reference=reference)
        with self.assertRaises(CameraError):
            self.observe(beam_depth(-.03), reference=reference)

    def test_outward_sign_is_selected_from_fixed_jaw(self):
        """카메라 영상의 좌우 부호와 무관하게 고정턱이 있는 쪽을 선택한다."""
        observed = self.observe(beam_depth(.01), hint=np.array([-1., 0., 0.]))
        self.assertLess(observed.outward[0], -.999)
        self.assertAlmostEqual(observed.edge_point_m[0], -.025, delta=.0015)


if __name__ == "__main__":
    unittest.main()
