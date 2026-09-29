"""빔 상대 삽입 자세를 IK로 생성하고 메시 정답으로 경계 양쪽 자료를 편성한다.
빔 길이 방향 구간과 경로 묶음을 분리하여 학습·검증·시험 간 누출을 막는다.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from common import REPO

sys.path.insert(0, str(REPO))
from kinematics.ik import IKSettings, InverseKinematics


X_INTERVALS = {
    "train": ((.14, .18), (.22, .26)),
    "validation": ((.182, .188), (.212, .218)),
    "test": ((.192, .208),),
}
LATERAL_INTERVALS = ((.032, .047), (.020, .032), (.004, .020), (-.003, .004))
PHASE_NAMES = ("before", "edge", "partial", "inserted")


def initialize_ik(trace_path):
    """프로세스별로 기존 순수 기구학 솔버와 이론 삽입 기준 자세를 준비한다."""
    global _solver, _reference_q, _reference_pose, _anchor_inverse
    _solver = InverseKinematics(settings=IKSettings(starts=1, max_iterations=100,
                                                  position_tolerance_m=1e-5,
                                                  rotation_tolerance_rad=1e-4))
    _reference_q = np.load(trace_path)["q"][-1, :7].copy()
    reference = _solver.fk.forward(_reference_q)
    _reference_pose = reference.T_world_tip_R.copy()
    _anchor_inverse = np.linalg.inv(reference.T_world_tip_L)


def generate_family(job):
    r"""하나의 삽입 경로 묶음에서 위치·높이·방향·턱 각도를 바꾸며 IK를 푼다.

    $$T_{LR}^{goal}=T_{WL}^{-1}T_{WR}^{goal}$$

    경로 묶음은 빔 길이 방향 위치와 기본 기울기를 공유한다. 충돌 여부로 IK를 거부하지 않는다.
    """
    split, family, count, seed = job
    rng = np.random.default_rng(seed)
    intervals = X_INTERVALS[split]
    x = rng.uniform(*intervals[family % len(intervals)])
    angle_range = .75 if (family // len(intervals)) % 2 == 0 else 3.
    angles = rng.uniform(-angle_range, angle_range, 3)
    rows = []
    failures = 0
    for index in range(count):
        phase = index % 4
        target = _reference_pose.copy()
        target[0, 3] = x
        target[1, 3] += rng.uniform(*LATERAL_INTERVALS[phase])
        if phase >= 2 and rng.random() < .5:
            target[2, 3] += rng.uniform(-.001, .003)
        else:
            target[2, 3] += rng.uniform(-.012 if phase < 2 else -.006, .010)
        target[:3, :3] = Rotation.from_euler("xyz", angles + rng.uniform(-.25, .25, 3), degrees=True).as_matrix()
        # 왼쪽 고정 팁을 기준으로 목표 자세 변환: $$T_{LR}^{goal}=T_{WL}^{-1}T_{WR}^{goal}$$
        relative = _anchor_inverse @ target
        seed_q = _reference_q + rng.normal(0, .08, 7)
        limits = _solver.fk.joint_limits
        if np.any((seed_q < limits[:, 0]) | (seed_q > limits[:, 1])):
            seed_q = _reference_q.copy()
        result = _solver.solve(relative, seed_q)
        if not result.success:
            failures += 1
            continue
        jaw = rng.uniform(-122., -116.) if rng.random() < .8 else rng.uniform(-116., -100.)
        q = np.r_[result.q_rad, np.deg2rad(jaw)]
        family_id = family + ("train", "validation", "test").index(split) * 100000
        rows.append((q, phase, family_id, target[:3, 3],
                     result.position_error_m, result.rotation_error_rad))
    return rows, failures


def collect_families(iterator, total, progress):
    """경로 묶음 결과를 순서대로 모으고 긴 자료 생성의 실제 진행량을 출력한다."""
    results = []
    for completed, result in enumerate(iterator, 1):
        results.append(result)
        if progress and (completed % 20 == 0 or completed == total):
            print(f"IK 경로 묶음 완료: {completed}/{total}", flush=True)
    return results


def candidate_pool(split, families, per_family, seed, trace_path, workers, *, progress=False):
    """경로 묶음마다 독립 난수를 주어 병렬 실행 순서와 무관한 후보를 생성한다."""
    jobs = [(split, i, per_family, seed + i * 1009) for i in range(families)]
    if workers == 1:
        initialize_ik(str(trace_path))
        results = collect_families(map(generate_family, jobs), families, progress)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_ik, initargs=(str(trace_path),)) as pool:
            results = collect_families(pool.map(generate_family, jobs), families, progress)
    rows = [row for part, _ in results for row in part]
    if not rows:
        raise RuntimeError("삽입 후보 IK가 모두 실패했습니다.")
    q, phase, family, position, position_error, rotation_error = zip(*rows)
    return {"q": np.asarray(q, dtype=np.float32), "phase": np.asarray(phase),
            "family": np.asarray(family), "target_position": np.asarray(position),
            "position_error_m": np.asarray(position_error), "rotation_error_rad": np.asarray(rotation_error)}, {
                "attempted": families * per_family, "ik_failed": sum(n for _, n in results),
                "families": families, "x_intervals_m": X_INTERVALS[split]}


def select_training(pool, count, rng):
    """단계별 비율과 실제 거리 ±5 mm의 안전·충돌 균형을 정답 기준으로 맞춘다."""
    if count < 60 or count % 60:
        raise ValueError("정확한 단계·거리 배분을 위해 표본 수는 60의 배수여야 합니다.")
    selected = []
    for phase in range(4):
        mask = pool["phase"] == phase
        groups = [(mask, count // 10)] if phase == 0 else [
            (mask & (pool["d"] > 0) & (pool["d"] <= .005), count * 7 // 60),
            (mask & (pool["d"] <= 0) & (pool["d"] >= -.005), count * 7 // 60),
            (mask & (np.abs(pool["d"]) > .005), count // 15),
        ]
        for eligible, quota in groups:
            indices = np.flatnonzero(eligible)
            if len(indices) < quota:
                raise ValueError(f"삽입 단계 {phase} 후보 부족: 필요 {quota}, 보유 {len(indices)}")
            selected.extend(rng.choice(indices, quota, replace=False).tolist())
    indices = rng.permutation(selected)
    return {key: value[indices] for key, value in pool.items()}


def distribution(pool):
    """거리·단계·경로 묶음과 IK 오차 분포를 수치로 남긴다."""
    d = pool["d"]
    return {"count": len(d), "collision_count": int((d <= 0).sum()),
            "near_5mm_count": int((np.abs(d) <= .005).sum()),
            "near_1mm_count": int((np.abs(d) <= .001).sum()),
            "phase_counts": {name: int((pool["phase"] == i).sum()) for i, name in enumerate(PHASE_NAMES)},
            "family_count": len(np.unique(pool["family"])),
            "max_ik_position_error_m": float(pool["position_error_m"].max()),
            "max_ik_rotation_error_rad": float(pool["rotation_error_rad"].max())}
