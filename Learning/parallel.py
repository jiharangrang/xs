"""독립 MuJoCo 프로세스를 유지하며 메시 거리 정답을 순서대로 병렬 계산한다."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from scene import DistanceScene


def initialize_worker(directory):
    """작업 프로세스마다 독립적인 거리 장면을 한 번 준비한다."""
    global _worker_scene
    _worker_scene = DistanceScene(directory)


def label_chunk(q):
    """현재 작업 프로세스의 장면으로 한 묶음의 정답을 계산한다."""
    return _worker_scene.label(q)


class ParallelLabeler:
    """여러 학습 회차에 걸쳐 재사용하는 거리 계산 프로세스 풀이다."""

    def __init__(self, scene, workers=1):
        """직렬 장면과 제한된 작업 프로세스 수를 보관한다."""
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
            raise ValueError("거리 계산 프로세스는 1~8개로 지정하세요.")
        self.scene = scene
        self.workers = workers
        self.pool = None

    def __enter__(self):
        """CUDA 상태를 상속하지 않는 spawn 방식으로 작업 풀을 준비한다."""
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker, initargs=(str(self.scene.directory),))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """성공과 예외 모두에서 작업 프로세스를 정리한다."""
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def label(self, q, *, progress=False):
        """입력 순서를 유지하며 각 묶음의 거리 결과를 합친다."""
        if self.workers == 1:
            return self.scene.label(q, progress=progress)
        if self.pool is None:
            raise RuntimeError("병렬 라벨러는 with 문 안에서 사용해야 합니다.")
        chunk_size = min(1024, max(1, (len(q) + self.workers - 1) // self.workers))
        chunks = [q[i:i + chunk_size] for i in range(0, len(q), chunk_size)]
        results = []
        completed = 0
        for values in self.pool.map(label_chunk, chunks):
            results.append(values)
            completed += len(values)
            if progress and (completed % 16384 == 0 or completed == len(q)):
                print(f"병렬 거리 라벨: {completed}/{len(q)}", flush=True)
        return np.concatenate(results) if results else np.empty(0, dtype=np.float32)
