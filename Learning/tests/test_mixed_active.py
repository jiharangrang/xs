"""삽입 시작 MCMC와 자료 교체 시 고정 비율·보존되는 자료의 계약을 검사한다."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active import boundary_samples
from mixed_active import assemble_dataset, sample_active, size_settings


class ConstantDistance(nn.Module):
    """시작 자세 전달 여부를 검사하기 위한 일정한 거리 함수다."""

    def forward(self, q):
        """모든 자세에 같은 거리 값을 반환한다."""
        return torch.zeros(len(q), dtype=q.dtype, device=q.device)


class MixedActiveTests(unittest.TestCase):
    """새 초기화 방식이 기존 추출과 자료 보호 조건을 깨뜨리지 않는지 확인한다."""

    def test_million_preserves_ratios_and_expands_families(self):
        """100만 실행이 비율·배치·독립 경로 묶음을 모두 확대하는지 확인한다."""
        small = size_settings(100000)
        large = size_settings(1000000)
        self.assertEqual(large["counts"], {"uniform": 350000, "active": 350000, "insertion": 300000})
        self.assertEqual(sum(large["counts"].values()), 1000000)
        self.assertEqual((small["train_families"], large["train_families"]), (120, 1200))
        self.assertEqual((small["batch_size"], large["batch_size"]), (1000, 10000))
        self.assertEqual(small["profile"], "mixed_100k_scratch")
        self.assertEqual(large["profile"], "mixed_1m_scratch")
        with self.assertRaises(ValueError):
            size_settings(100001)

    def test_million_active_replacement_size(self):
        """대량 갱신에서 전역·삽입 시작 MCMC가 합쳐 35만 개를 만드는지 확인한다."""
        limits = np.tile([-1., 1.], (8, 1))
        task_q = np.zeros((300000, 8), dtype=np.float32)
        proposed = [(np.zeros((175000, 8), dtype=np.float32), {}),
                    (np.ones((175000, 8), dtype=np.float32), {})]
        with patch("mixed_active.boundary_samples", side_effect=proposed) as sampler:
            q, info = sample_active(None, SimpleNamespace(limits=limits), task_q,
                                    np.random.default_rng(9), "cpu", count=350000, steps=5)
        self.assertEqual(q.shape, (350000, 8))
        self.assertEqual((info["global_count"], info["insertion_seeded_count"]), (175000, 175000))
        self.assertEqual(sampler.call_args_list[0].args[2], 175000)
        self.assertEqual(sampler.call_args_list[1].kwargs["initial_q"].shape, (175000, 8))
        self.assertTrue(np.all(q[:175000] == 0))
        self.assertTrue(np.all(q[175000:] == 1))

    def check_seeded_sampler(self, device):
        """아주 작은 제안에서 지정 시작 자세가 유지되고 입력 배열이 보존되는지 검사한다."""
        initial = np.full((16, 2), .6, dtype=np.float32)
        original = initial.copy()
        limits = np.array([[-1., 1.], [-1., 1.]])
        result, info = boundary_samples(ConstantDistance().to(device), limits, 16, np.random.default_rng(12),
                                       steps=3, proposal_std=1e-10, device=device, initial_q=initial, chunk=5)
        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(initial, original)
        self.assertEqual(info["acceptance_fraction"], 1.)
        for invalid in (np.ones((15, 2)), np.full((16, 2), np.nan), np.full((16, 2), 2.)):
            with self.assertRaises(ValueError):
                boundary_samples(ConstantDistance().to(device), limits, 16, np.random.default_rng(12),
                                 steps=1, device=device, initial_q=invalid)

    def test_seeded_cpu(self):
        """CPU 추출의 시작점 전달과 입력 검사를 확인한다."""
        self.check_seeded_sampler("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 서버에서 실행")
    def test_seeded_cuda(self):
        """CUDA 추출의 시작점 전달과 입력 검사를 확인한다."""
        self.check_seeded_sampler("cuda")

    def test_refresh_preserves_global_and_insertion_data(self):
        """능동 자료를 교체해도 균등·삽입 자료와 세 묶음의 비율이 보존되는지 확인한다."""
        counts = {"uniform": 35, "active": 35, "insertion": 30}
        rng = np.random.default_rng(13)
        parts = {name: (rng.normal(size=(count, 8)).astype(np.float32), rng.normal(size=count).astype(np.float32))
                 for name, count in counts.items()}
        before_q, before_d, source = assemble_dataset(parts, counts)
        parts["active"] = np.full((35, 8), 10., np.float32), np.full(35, -10., np.float32)
        after_q, after_d, after_source = assemble_dataset(parts, counts)
        np.testing.assert_array_equal(source, after_source)
        self.assertEqual(np.bincount(source).tolist(), [35, 35, 30])
        np.testing.assert_array_equal(before_q[source != 1], after_q[source != 1])
        np.testing.assert_array_equal(before_d[source != 1], after_d[source != 1])
        self.assertTrue(np.all(after_d[source == 1] == -10.))
        parts["active"] = np.zeros((34, 8)), np.zeros(34)
        with self.assertRaises(ValueError):
            assemble_dataset(parts, counts)


if __name__ == "__main__":
    unittest.main()
