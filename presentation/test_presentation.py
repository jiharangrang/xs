"""발표 시간 매핑과 고정점 연속성 및 실제 기록 경계를 검사한다."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from presentation.choreography import Choreography, PresentationSettings
from presentation.editing import Cut, EditTimeline, build_edit
from presentation.recording import Recording
from presentation.video import VideoClip

ROOT = Path(__file__).resolve().parents[1] / 'outputs/walk_recordings/20260919_144803_kst_two_cycles'


class EditingTests(unittest.TestCase):
    """컷 경계와 배속이 공통 로그 시각에 정확히 연결되는지 검사한다."""

    def test_gap_and_speed(self):
        """출력 공백 없이 다음 원본 구간으로 이동하는지 확인한다."""
        edit = EditTimeline([Cut(0, 10, 16, 3), Cut(1, 40, 42, 2)])
        self.assertEqual(edit.duration, 3)
        self.assertEqual(edit.sample(1)[0], 13)
        self.assertEqual(edit.sample(2)[0], 40)
        self.assertEqual(edit.sample(3)[0], 42)

    def test_round_trip(self):
        """저장 후 다시 읽은 편집표가 같은 프레임 경계를 유지하는지 확인한다."""
        edit = EditTimeline([Cut(0, -.4, 1.234, 3)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'edit.json'
            edit.save(path)
            restored = EditTimeline.load(path)
        self.assertEqual(edit.frames, restored.frames)
        self.assertEqual(edit.sample(.4), restored.sample(.4))

    def test_invalid_inputs(self):
        """음수 배속과 비유한 시각을 거부하는지 확인한다."""
        with self.assertRaises(ValueError):
            EditTimeline([Cut(0, 1, 2, -1)])
        with self.assertRaises(ValueError):
            PresentationSettings(sag_mm=float('nan'))


@unittest.skipUnless(ROOT.exists(), '실제 녹화 묶음이 있어야 실행하는 통합 검사')
class RecordingTests(unittest.TestCase):
    """실제 녹화의 재구성 경로가 지지점과 원본 시간을 유지하는지 검사한다."""

    @classmethod
    def setUpClass(cls):
        """전체 녹화를 한 번만 읽고 경로를 준비한다."""
        cls.recording = Recording(ROOT)
        cls.path = Choreography(cls.recording)

    def test_original_stop_preserved(self):
        """원본의 중지 상태를 성공으로 바꾸지 않는지 확인한다."""
        self.assertEqual(self.recording.status(-34.0), ('STAGE 3', 'STOPPED'))

    def test_grasp_and_final_progress(self):
        """삽입 완료와 반복 보행의 도착 팁 위치가 이론값과 맞는지 확인한다."""
        for seconds, left_x, right_x in [(0, 0, .2), (244, .15, .3)]:
            q, anchor, _ = self.path.sample(seconds)
            pose = anchor.place(self.path.solver.fk.forward(q))
            self.assertAlmostEqual(pose.T_world_tip_L[0, 3], left_x, places=6)
            self.assertAlmostEqual(pose.T_world_tip_R[0, 3], right_x, places=6)
            self.assertAlmostEqual(pose.T_world_tip_L[2, 3], pose.T_world_tip_R[2, 3], places=6)

    def test_world_continuity(self):
        """고정 발을 바꾸는 순간에도 양쪽 발의 위치가 연속인지 확인한다."""
        fk = self.path.solver.fk
        for previous, current in zip(self.path.motions, self.path.motions[1:]):
            before = previous.anchor.place(fk.forward(previous.q[-1]))
            after = current.anchor.place(fk.forward(current.q[0]))
            np.testing.assert_allclose(before.T_world_tip_L, after.T_world_tip_L, atol=1e-6)
            np.testing.assert_allclose(before.T_world_tip_R, after.T_world_tip_R, atol=1e-6)
            self.assertLessEqual(previous.end, current.start)

    def test_observation_holds(self):
        """보정 이동 사이의 관측 시간에 몸통이 계속 움직이지 않는지 확인한다."""
        for previous, current in zip(self.path.motions, self.path.motions[1:]):
            if current.start > previous.end:
                midpoint = (current.start + previous.end) / 2
                np.testing.assert_allclose(self.path.sample(midpoint)[0], previous.q[-1])

    def test_gripper_timing(self):
        """실제 열림 구간이 끝날 때 모델도 완전히 열리는지 확인한다."""
        run = next(run for run in self.recording.runs if run['stage'] == 8)
        phase = next(phase for phase in run['phases'] if phase['phase'] == 'advancing')
        seconds = self.recording.relative(phase['monotonic_s'])
        self.assertAlmostEqual(np.rad2deg(self.path.sample(seconds)[2]['G_R']), -120)

    def test_random_seek_is_deterministic(self):
        """앞뒤 탐색 순서에 따라 동일 프레임의 자세가 변하지 않는지 확인한다."""
        first = self.path.sample(72)[0]
        self.path.sample(220)
        self.path.sample(-50)
        np.testing.assert_array_equal(first, self.path.sample(72)[0])

    def test_edited_coverage(self):
        """동작마다 영상이 존재하고 긴 촬영 공백이 편집에서 빠지는지 확인한다."""
        clips = [VideoClip(Path('one.mp4'), -74.071218, 242.408833, 59.94), VideoClip(Path('two.mp4'), 195.928782, 44.044, 59.94)]
        edit = build_edit(self.path, clips)
        for cut in edit.cuts:
            clip = clips[cut.clip]
            self.assertGreaterEqual(cut.start, clip.start)
            self.assertLessEqual(cut.end, clip.start + clip.duration)
        self.assertLess(edit.duration, 100)
        self.assertTrue(any(cut.clip == 1 for cut in edit.cuts))
        self.assertFalse(any(cut.start < 180 < cut.end for cut in edit.cuts))


if __name__ == '__main__':
    unittest.main()
