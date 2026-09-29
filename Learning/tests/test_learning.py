"""학습 입력·거리 정답·능동 추출·경로 비교의 핵심 계약을 검증한다."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch
from torch import nn

from active import boundary_samples, replace_random
from common import ARTIFACTS, BODIES, JOINTS, MODEL, signature
from model import LinkFeatures, SE3NN, load_checkpoint, pose_matrix
from scene import DistanceScene
from train import TrainConfig, metrics, profile_config
from support import original_beam_pose, require_initial_grasp


class CoreTests(unittest.TestCase):
    """기하 전처리와 학습 결과 없이도 검사할 수 있는 논문 구현 계약이다."""

    @classmethod
    def setUpClass(cls):
        """기존 XML의 순기구학을 독립 기준으로 준비한다."""
        torch.set_num_threads(1)
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))

    def test_se3_features_match_mujoco_and_jacobian(self):
        """임의 자세의 링크 변환과 관절 미분을 MuJoCo 및 유한차분과 대조한다."""
        features = LinkFeatures(self.model).double()
        rng = np.random.default_rng(94)
        qs = rng.uniform(-1, 1, (12, 8))
        data = mujoco.MjData(self.model)
        expected = []
        for q in qs:
            for name, value in zip(JOINTS, q, strict=True):
                data.joint(name).qpos[0] = value
            mujoco.mj_kinematics(self.model, data)
            beam = data.body("ibeam")
            inverse = np.linalg.inv(pose_matrix(beam.xpos, self.model.body("ibeam").quat))
            row = []
            for name in BODIES:
                body = data.body(name)
                transform = np.eye(4)
                transform[:3, :3] = body.xmat.reshape(3, 3)
                transform[:3, 3] = body.xpos
                row.extend((inverse @ transform)[:3].ravel())
            expected.append(row)
        actual = features(torch.tensor(qs, dtype=torch.float64))
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-7)
        q = torch.tensor(qs[:1], requires_grad=True, dtype=torch.float64)
        weights = torch.linspace(.1, 1, features.output_dim, dtype=torch.float64)
        gradient = torch.autograd.grad((features(q) * weights).sum(), q)[0].detach().numpy()[0]
        finite = []
        for i in range(8):
            delta = np.zeros_like(qs[:1]); delta[0, i] = 1e-5
            plus = (features(torch.tensor(qs[:1] + delta)) * weights).sum().item()
            minus = (features(torch.tensor(qs[:1] - delta)) * weights).sum().item()
            finite.append((plus - minus) / 2e-5)
        np.testing.assert_allclose(gradient, finite, atol=1e-7)

    def test_paper_network_architecture(self):
        """네 은닉층과 단일 거리 출력이 논문 구조와 같은지 확인한다."""
        model = SE3NN(self.model)
        layers = [item for item in model.network if isinstance(item, nn.Linear)]
        self.assertEqual([layer.out_features for layer in layers], [128, 128, 128, 128, 1])
        self.assertEqual(layers[0].in_features, 96)

    def test_fixed_grasp_rejects_moved_beam(self):
        """빔만 옮겨 고정 파지를 깨뜨리는 과거 배치 오류를 거부한다."""
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)
        require_initial_grasp(model, data)
        model.body("ibeam").pos[2] += .01
        with self.assertRaisesRegex(ValueError, "파지 상태"):
            require_initial_grasp(model, data)
        model.body("ibeam").pos[:] = original_beam_pose()["pos"]
        data.joint("G_L").qpos[0] = -.1
        with self.assertRaisesRegex(ValueError, "파지 상태"):
            require_initial_grasp(model, data)

    def test_native_signed_distance_known_boxes(self):
        """명시적 거리 질의는 접촉 비활성 상태에서도 간격과 관통의 부호를 구분한다."""
        xml = '<mujoco><worldbody><geom name="a" type="box" size=".1 .1 .1" contype="0" conaffinity="0"/><body name="b" pos=".3 0 0"><geom name="b" type="box" size=".1 .1 .1" contype="0" conaffinity="0"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        self.assertAlmostEqual(mujoco.mj_geomDistance(model, data, 0, 1, 10, None), .1)
        model.body("b").pos[0] = .15
        mujoco.mj_kinematics(model, data)
        self.assertAlmostEqual(mujoco.mj_geomDistance(model, data, 0, 1, 10, None), -.05)

    def test_boundary_sampling_concentrates_without_clipping(self):
        """정답 경계가 알려진 함수에서 표본이 경계로 모이고 관절 범위를 지키는지 확인한다."""
        class Plane(nn.Module):
            """첫 관절의 영점을 충돌 경계로 사용하는 시험용 함수다."""

            def forward(self, q):
                """첫 관절값을 거리로 반환한다."""
                return q[:, 0]

        limits = np.array([[-1., 1.], [-1., 1.]])
        q, info = boundary_samples(Plane(), limits, 256, np.random.default_rng(7),
                                   sigma_e=.05, proposal_std=.15, steps=100)
        self.assertLess(np.mean(np.abs(q[:, 0])), .06)
        self.assertTrue(np.all((q >= -1) & (q <= 1)))
        self.assertGreater(info["acceptance_fraction"], 0)

    def test_fixed_dataset_replacement(self):
        """새 정답을 교체해도 크기가 고정되고 교체하지 않은 표본은 유지되는지 확인한다."""
        q = np.zeros((20, 8), np.float32); labels = np.zeros(20, np.float32)
        indices = replace_random(q, labels, np.ones((5, 8)), np.ones(5), np.random.default_rng(3))
        self.assertEqual(q.shape, (20, 8))
        self.assertEqual(len(set(indices)), 5)
        self.assertEqual(float(labels.sum()), 5)
        np.testing.assert_array_equal(q[indices], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 서버에서 실행")
    def test_cuda_boundary_sampling(self):
        """GPU에서도 경계 표본이 영점에 모이고 관절 제한을 지키는지 확인한다."""
        class Plane(nn.Module):
            """첫 관절을 부호 있는 거리로 사용하는 CUDA 시험 함수다."""

            def forward(self, q):
                """첫 관절값을 거리로 반환한다."""
                return q[:, 0]

        limits = np.array([[-1., 1.], [-1., 1.]])
        q, info = boundary_samples(Plane().cuda(), limits, 256, np.random.default_rng(7),
                                  sigma_e=.05, proposal_std=.15, steps=100, device="cuda")
        self.assertLess(np.mean(np.abs(q[:, 0])), .06)
        self.assertTrue(np.all((q >= -1) & (q <= 1)))
        self.assertGreater(info["acceptance_fraction"], 0)

    def test_metrics_do_not_hide_missed_collisions(self):
        """충돌을 안전하다고 예측한 오류와 음성 표본만 있는 경우를 구분한다."""
        result = metrics(np.array([.001, -.002, .2]), np.array([-.001, -.003, .2]))
        self.assertEqual(result["missed_collision_count"], 1)
        self.assertEqual(result["false_safe_rate"], .5)
        self.assertEqual(result["near_count"], 2)
        self.assertIsNone(metrics(np.ones(2), np.ones(2))["false_safe_rate"])

    def test_paper_auroc_and_near_band(self):
        """논문의 경계 범위와 AUROC의 점수 방향·동점·단일 클래스 처리를 확인한다."""
        truth = np.array([-.2, -.05, .05, .2])
        correct = metrics(truth, truth)
        self.assertEqual(correct["near_m"], .1)
        self.assertEqual(correct["near_count"], 2)
        self.assertEqual(correct["auroc"], 1.)
        self.assertEqual(correct["near_auroc"], 1.)
        self.assertEqual(correct["near_accuracy"], 1.)
        self.assertEqual(metrics(-truth, truth)["auroc"], 0.)
        self.assertEqual(metrics(np.zeros(4), truth)["auroc"], .5)
        self.assertIsNone(metrics(np.ones(2), np.ones(2))["auroc"])
        self.assertIsNone(metrics(np.ones(2), np.ones(2))["near_auroc"])

    def test_smoke_is_default_and_full_is_explicit(self):
        """기본 예산은 소규모이고 대량 설정은 실행하지 않고 값만 확인한다."""
        smoke, full = profile_config("smoke"), profile_config("full")
        smoke.validate(); full.validate()
        self.assertLessEqual(smoke.train_size, 2048)
        self.assertEqual(full.train_size, 1_000_000)
        self.assertEqual((full.epochs, full.training_rounds, full.replace_size, full.mcmc_steps), (1000, 20, 100_000, 1000))
        self.assertEqual(full.training_rounds * full.epochs, 20_000)
        self.assertEqual(full.near_m, .1)
        with self.assertRaises(ValueError):
            TrainConfig(replace_size=9999).validate()


@unittest.skipUnless((ARTIFACTS / "scene/scene.json").exists(), "장면 전처리 후 실행")
class SceneTests(unittest.TestCase):
    """실제 CAD 조각과 저장된 6단계 경로에 대한 통합 검사다."""

    @classmethod
    def setUpClass(cls):
        """완료된 장면과 경로를 읽기 전용으로 준비한다."""
        cls.scene = DistanceScene()
        with np.load(ARTIFACTS / "scene/trace.npz") as trace:
            cls.q = trace["q"].copy()
            cls.stages = trace["stage"].copy()

    def test_all_six_stages_and_joint_limits(self):
        """이론 경로가 모든 단계를 포함하고 현재 관절 제한을 만족하는지 확인한다."""
        self.assertEqual(set(self.stages), set(range(1, 7)))
        self.assertTrue(np.all(self.q >= self.scene.limits[:, 0] - 1e-7))
        self.assertTrue(np.all(self.q <= self.scene.limits[:, 1] + 1e-7))

    def test_scene_is_portable_without_original_paths(self):
        """생성 컴퓨터의 절대 경로가 없어도 현재 원본의 해시를 검사하며 장면을 읽는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ARTIFACTS / "scene/scene.json").read_text())
            for info in config["parts"].values():
                info["path"] = "/unavailable/original/source.stl"
            config["scene_id"] = signature({k: v for k, v in config.items() if k != "scene_id"})
            (root / "scene.json").write_text(json.dumps(config))
            (root / "geometry.npz").symlink_to(ARTIFACTS / "scene/geometry.npz")
            moved = DistanceScene(root)
            self.assertAlmostEqual(moved.distance(self.q[-1]), self.scene.distance(self.q[-1]), places=9)

    def test_parallel_labels_equal_serial(self):
        """여러 MuJoCo 프로세스의 결과와 입력 순서가 직렬 계산과 일치한다."""
        from parallel import ParallelLabeler
        rng = np.random.default_rng(64)
        q = rng.uniform(self.scene.limits[:, 0], self.scene.limits[:, 1], (32, 8))
        with ParallelLabeler(self.scene, workers=2) as labeler:
            actual = labeler.label(q)
        np.testing.assert_allclose(actual, self.scene.label(q), atol=1e-8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 서버에서 실행")
    def test_cpu_cuda_predictions_match(self):
        """같은 가중치의 CPU와 CUDA 순기구학·거리 추론 결과가 일치한다."""
        model = SE3NN(self.scene.model).eval()
        q = torch.tensor(self.q[::15], dtype=torch.float32)
        with torch.no_grad():
            expected = model(q).numpy()
            actual = model.cuda()(q.cuda()).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_ideal_initial_pose_and_fixed_grasp(self):
        """실측 보정 없이 영점 초기 자세에서 출발하며 전체 경로의 왼쪽 파지가 유지된다."""
        metadata = json.loads((ARTIFACTS / "scene/trace.json").read_text())
        self.assertFalse(metadata["measured_logs_used"])
        np.testing.assert_array_equal(self.q[0, :7], 0)
        np.testing.assert_allclose(self.q[:, 7], np.deg2rad(-120.))
        for key in ("pos", "quat"):
            np.testing.assert_allclose(self.scene.config["beam_pose"][key], original_beam_pose()[key])
        for q in self.q:
            self.scene.set_q(q)
            require_initial_grasp(self.scene.model, self.scene.data)

    def test_ideal_path_subsamples_do_not_penetrate(self):
        """경로의 사분점까지 검사한 메시 거리와 고정 파지 검증 결과를 확인한다."""
        from support import check_ideal_path
        q, distances, report = check_ideal_path(self.scene, self.q)
        self.assertEqual(len(q), (len(self.q) - 1) * 4 + 1)
        self.assertTrue(report["support_valid_all_samples"])
        self.assertGreater(distances.min(), 0)

    def test_pruning_equals_all_pairs(self):
        """외접 구 가지치기의 결과가 모든 메시 조각 쌍 계산과 같은지 확인한다."""
        rng = np.random.default_rng(11)
        random_q = rng.uniform(self.scene.limits[:, 0], self.scene.limits[:, 1], (8, 8))
        for q in np.vstack((self.q[::150], random_q)):
            actual = self.scene.distance(q)
            expected = min(mujoco.mj_geomDistance(self.scene.model, self.scene.data, a, b, 10., None)
                           for a in self.scene.robot_geoms for b in self.scene.beam_geoms)
            self.assertAlmostEqual(actual, expected, places=9)

    def test_ideal_scene_features_match_mujoco(self):
        """이론 배치에서 신경망의 링크 좌표와 거리 장면의 좌표가 일치한다."""
        features = LinkFeatures(self.scene.model)
        q = self.q[-1]
        self.scene.set_q(q)
        beam = self.scene.data.body("ibeam")
        inverse = np.linalg.inv(pose_matrix(beam.xpos, self.scene.model.body("ibeam").quat))
        expected = []
        for name in BODIES:
            body = self.scene.data.body(name)
            pose = np.eye(4); pose[:3, :3] = body.xmat.reshape(3, 3); pose[:3, 3] = body.xpos
            expected.extend((inverse @ pose)[:3].ravel())
        np.testing.assert_allclose(features(torch.tensor(q[None], dtype=torch.float32)).detach().numpy()[0], expected, atol=2e-7)

    def test_checkpoint_scene_mismatch_rejected(self):
        """다른 기하·배치에 대한 모델을 비교에 섞지 못하도록 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"scene_id": "different", "joint_names": list(JOINTS)}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, self.scene)


@unittest.skipUnless((ARTIFACTS / "smoke/training.json").exists(), "소규모 학습 후 실행")
class SmokeResultTests(unittest.TestCase):
    """한 번 실행한 작은 학습의 저장·재사용·비교 결과를 검증한다."""

    def test_training_updates_and_held_out_data(self):
        """능동 반복의 자료 크기가 고정되고 독립 검증 손실이 줄었는지 확인한다."""
        report = json.loads((ARTIFACTS / "smoke/training.json").read_text())
        self.assertEqual(report["profile"], "smoke")
        active = report["methods"]["active"]
        uniform = report["methods"]["uniform"]
        self.assertTrue(active["validation_mse_reduced"])
        self.assertEqual(active["initial_validation"], uniform["initial_validation"])
        self.assertEqual([row["dataset_count"] for row in active["rounds"]], [2048] * 3)
        self.assertEqual([row["replacement_count"] for row in active["rounds"][:-1]], [256, 256])
        with np.load(ARTIFACTS / "smoke/dataset.npz") as data:
            train_rows = {row.tobytes() for row in data["train_q"]}
            self.assertFalse(any(row.tobytes() in train_rows for row in data["test_q"]))
            self.assertFalse(any(row.tobytes() in train_rows for row in data["validation_q"]))

    def test_reload_reproduces_saved_evaluation(self):
        """저장 가중치를 다시 읽어도 독립 시험의 같은 거리 예측을 만드는지 확인한다."""
        from train import predict
        scene = DistanceScene()
        model, checkpoint = load_checkpoint(ARTIFACTS / "smoke/active.pt", scene)
        with np.load(ARTIFACTS / "smoke/dataset.npz") as data:
            result = metrics(predict(model, data["test_q"]), data["test_d"])
        saved = json.loads((ARTIFACTS / "smoke/training.json").read_text())["methods"]["active"]["test"]
        self.assertAlmostEqual(result["mae_m"], saved["mae_m"], places=6)
        self.assertFalse(checkpoint["qualified"])

    def test_comparison_contains_all_stages_and_errors(self):
        """시간 비교가 같은 경로를 쓰고 충돌 오류도 기록하는지 확인한다."""
        report = json.loads((ARTIFACTS / "comparison/comparison.json").read_text())
        self.assertEqual(set(report["stages"]), {str(i) for i in range(1, 7)})
        with np.load(ARTIFACTS / "scene/trace.npz") as trace:
            self.assertEqual(report["samples"], len(trace["q"]))
        self.assertFalse(report["qualified"])
        self.assertIn("false_safe_rate", report["error"])
        for name in ("mesh", "learned"):
            self.assertGreater(report["timing"][name]["mean_ms"], 0)


if __name__ == "__main__":
    unittest.main()
