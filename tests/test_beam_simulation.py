"""실제 로봇 CAD의 서로 다른 경로 자세에서 빔 추정을 정답과 비교한다."""

from pathlib import Path
import unittest

import numpy as np

from perception.beam import estimate_beam
from planning.motion_path import load_path
from scripts.compare_beam import reference_errors
from simulation.beam import beam_reference
from simulation.camera import JOINT_NAMES, render_camera
from simulation.model import load_model, set_path_time


class BeamSimulationTests(unittest.TestCase):
    """이상적인 깊이에서도 생길 수 있는 좌표·단위·고정점 오류를 확인한다."""

    def test_three_path_poses_against_cad_geometry(self):
        """경로의 시작·중간·끝 자세에서 같은 알고리즘이 CAD 폭과 방향을 복원한다."""
        path = load_path(Path(__file__).resolve().parents[1] / "outputs/rear_pull.json")
        for elapsed in (0, path.duration_s / 2, path.duration_s):
            with self.subTest(elapsed=elapsed):
                model, data = load_model()
                index = set_path_time(model, data, path, elapsed)
                angles = {name: float(np.rad2deg(data.joint(name).qpos[0])) for name in JOINT_NAMES}
                frame = render_camera(angles, include_beam=True, anchor=path.segments[index].anchor)
                reference = beam_reference(model, data)
                self.assertAlmostEqual(reference["width_m"], 0.070, delta=1e-6)
                estimate = estimate_beam(frame.depth_m, frame.camera_info["depth"])
                self.assertEqual(estimate.status, "two_edges")
                errors = reference_errors(estimate, reference)
                self.assertLess(errors["normal_error_deg"], 0.1)
                self.assertLess(errors["axis_error_deg"], 0.2)
                self.assertLess(errors["plane_distance_error_mm"], 0.1)
                self.assertLess(errors["width_error_mm"], 0.5)
                self.assertLess(errors["centerline_point_error_mm"], 0.5)


if __name__ == "__main__":
    unittest.main()
