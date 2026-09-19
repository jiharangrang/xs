"""새 프레임 장벽과 깊이 누락·모서리·시간 일관성에 대한 관측 처리를 검증한다."""

from dataclasses import replace
import time
import unittest

import numpy as np

from hardware.camera import CameraError, RGBDFrame
from perception.beam import BeamDetectionError, estimate_beam
from perception.beam_observation import ObservationSettings, combine_estimates, observe_beam
from test_beam import PROFILE, synthetic_beam


class FrameReader:
    """지정한 시각과 원본 깊이를 차례로 전달하는 카메라 대역이다."""

    def __init__(self, frames):
        """가짜 원본 프레임과 내부 파라미터를 준비한다."""
        self.frames = iter(frames)
        self.info = {"depth": PROFILE}

    def read(self, **kwargs):
        """준비한 프레임이 소진되면 새 관측이 없음을 알린다."""
        try:
            return next(self.frames)
        except StopIteration as error:
            raise CameraError("새 프레임 없음") from error


class BeamObservationTests(unittest.TestCase):
    """품질이 확인된 서로 다른 새 영상만 관측 완료로 인정하는지 확인한다."""

    def test_stale_duplicate_and_missing_frames_cannot_count_as_observation(self):
        """이동 전 영상과 중복·누락을 건너뛴 뒤 새 유효 프레임 묶음만 반환한다."""
        depth, truth = synthetic_beam(noisy=True)
        now = time.monotonic()
        stamps = [(now - 1., 1, depth), (now + .01, 2, depth), (now + .02, 2, depth),
                  (now + .03, 3, np.full_like(depth, np.nan)),
                  (now + .04, 4, depth), (now + .05, 5, depth), (now + .06, 6, depth)]
        frames = [RGBDFrame(np.zeros((2, 2, 3), np.uint8), d, seq, seq, stamp) for stamp, seq, d in stamps]
        result = observe_beam(FrameReader(frames), now, clock=lambda: now + .1)
        self.assertEqual(result.frames, 3)
        self.assertEqual(result.first_frame_s, now + .04)
        self.assertEqual(result.last_frame_s, now + .06)
        self.assertGreater(result.normal @ truth["normal"], .999)

    def test_plain_surface_is_not_accepted_as_beam(self):
        """평면만 있고 빔 폭을 확인하지 못하면 보정 입력으로 반환하지 않는다."""
        now = time.monotonic()
        frame = RGBDFrame(np.zeros((2, 2, 3), np.uint8), np.full((240, 320), .25), 1, 1, now + .01)
        with self.assertRaises(CameraError):
            observe_beam(FrameReader([frame]), now, clock=lambda: now + .1)

    def test_inconsistent_normals_and_cancelled_observation_are_rejected(self):
        """법선이 흔들리는 묶음과 중지된 관측은 완료 신호가 되지 않는다."""
        depth, _ = synthetic_beam()
        estimate = estimate_beam(depth, PROFILE)
        other = replace(estimate, normal=np.array([0., 0., -1.]))
        with self.assertRaises(BeamDetectionError):
            combine_estimates([estimate, other, estimate], [1, 2, 3], ObservationSettings())
        with self.assertRaises(BeamDetectionError):
            observe_beam(FrameReader([]), 0, cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
