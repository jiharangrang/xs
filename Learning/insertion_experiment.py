"""기존 능동 학습 모델에서 삽입 자료의 추가 효과를 같은 학습 예산으로 비교한다.
원본 모델을 보존하고 전역 시험·새 삽입 시험·기존 경로 회귀 결과를 분리한다.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from common import ARTIFACTS, JOINTS, REPO, file_hash, write_json
from insertion_data import LATERAL_INTERVALS, X_INTERVALS, candidate_pool, distribution, select_training
from model import load_checkpoint
from parallel import ParallelLabeler
from scene import DistanceScene
from train import TrainConfig, fit_block, metrics, predict


def detailed_metrics(predicted, truth):
    """안전 자세의 과검출과 충돌 누락을 구분하고 좁은 경계를 별도로 평가한다."""
    result = metrics(predicted, truth)
    result["false_collision_count"] = int(((truth > 0) & (predicted <= 0)).sum())
    result["safe_count"] = int((truth > 0).sum())
    result["bias_m"] = float(np.mean(predicted - truth))
    for name, width in (("near_1mm", .001), ("near_5mm", .005)):
        mask = np.abs(truth) <= width
        if mask.any():
            result[name] = metrics(predicted[mask], truth[mask])
            result[name]["false_collision_count"] = int(((truth[mask] > 0) & (predicted[mask] <= 0)).sum())
    return result


def old_training_subset(initial, active, count, rng):
    """기존 균등 자료와 능동 추출 자료에서 절반씩 중복 없이 뽑는다."""
    first = rng.choice(len(initial["train_q"]), count // 2, replace=False)
    second = rng.choice(len(active["q"]), count - count // 2, replace=False)
    return np.concatenate((initial["train_q"][first], active["q"][second])), np.concatenate((initial["train_d"][first], active["d"][second]))


def evaluate(model, datasets, output, name):
    """같은 평가 입력에서 모든 방법의 예측과 지표를 저장한다."""
    report = {}
    arrays = {}
    for split, (q, truth) in datasets.items():
        predicted = predict(model, q)
        report[split] = detailed_metrics(predicted, truth)
        arrays[split] = predicted
    np.savez_compressed(output / f"{name}_predictions.npz", **arrays)
    return report


def run(args):
    """자료 생성과 두 재학습을 실행하며 원본 가중치의 불변성을 확인한다."""
    if args.output.exists():
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다: {args.output}")
    if args.task_count % 60 or args.task_count < 60 or args.epochs < 1 or not 1 <= args.workers <= 8:
        raise ValueError("삽입 표본 수는 60의 배수, 학습 횟수는 양수, 작업자는 1~8개여야 합니다.")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    scene = DistanceScene()
    checkpoint_path = args.baseline / "active.pt"
    baseline_hash = file_hash(checkpoint_path)
    total = args.task_count * 10 // 3
    config = TrainConfig(train_size=total, epochs=args.epochs, training_rounds=1,
                         batch_size=min(1000, total), learning_rate=.0001, seed=71,
                         device=args.device, label_workers=args.workers)
    report = {"scene_id": scene.scene_id, "baseline_sha256": baseline_hash, "config": asdict(config),
              "interpretation": "능동 학습된 SE3NN의 자료 편성 대조 실험; 이번 추가 학습은 MCMC 갱신 없이 수행",
              "selection": "두 방법 모두 고정 횟수의 마지막 가중치 사용; 시험 점수로 선택하지 않음",
              "task_fraction": .3, "old_uniform_fraction_within_replay": .5, "data": {}, "methods": {}}
    report["sampling"] = {"x_intervals_world_m": X_INTERVALS, "lateral_offset_intervals_m": LATERAL_INTERVALS,
                          "family_rotation_deg": [.75, 3.], "rotation_jitter_deg": .25,
                          "height_offset_broad_m": [-.012, .010], "height_offset_inserted_m": [-.006, .010],
                          "height_offset_dense_m": [-.001, .003], "dense_height_fraction_in_inserted": .5,
                          "jaw_main_deg": [-122., -116.], "jaw_main_fraction": .8,
                          "jaw_other_deg": [-116., -100.], "seed_q_std_rad": .08,
                          "training_phase_fractions": [.1, .3, .3, .3],
                          "training_near_5mm_min_fraction": .7,
                          "evaluation_sampling": "후보 분포 그대로 평가; 정답으로 균형을 조절하지 않음"}
    report["source_sha256"] = {name: file_hash(REPO / name) for name in (
        "Learning/insertion_data.py", "Learning/insertion_experiment.py", "Learning/train.py",
        "Learning/scene.py", "Learning/model.py", "kinematics/ik.py", "kinematics/fk.py")}
    write_json(args.output / "config.json", report)
    pools = {}
    with ParallelLabeler(scene, args.workers) as labeler:
        for offset, (split, families, per_family) in enumerate((
                ("train", 120, args.candidates_per_family), ("validation", 20, 160), ("test", 32, 200))):
            print(f"삽입 {split} IK 후보 생성: {families * per_family}개", flush=True)
            pool, info = candidate_pool(split, families, per_family, 7100 + 100000 * offset,
                                        ARTIFACTS / "scene/trace.npz", args.workers)
            pool["d"] = labeler.label(pool["q"], progress=True)
            np.savez_compressed(args.output / f"{split}_candidates.npz", **pool)
            info["candidate_distribution"] = distribution(pool)
            if split == "train":
                pool = select_training(pool, args.task_count, np.random.default_rng(74))
            pools[split] = pool
            info["selected_distribution"] = distribution(pool)
            report["data"][split] = info
            np.savez_compressed(args.output / f"task_{split}.npz", **pool)
            write_json(args.output / "progress.json", report)
            print(f"삽입 {split}: {info['selected_distribution']}", flush=True)
    initial = dict(np.load(args.baseline / "dataset.npz"))
    active = dict(np.load(args.baseline / "active_final_dataset.npz"))
    old_q, old_d = old_training_subset(initial, active, total, np.random.default_rng(75))
    order = np.random.default_rng(76).permutation(total)
    old_q, old_d = old_q[order], old_d[order]
    mixed_q = np.concatenate((old_q[:-args.task_count], pools["train"]["q"]))
    mixed_d = np.concatenate((old_d[:-args.task_count], pools["train"]["d"]))
    trace = np.load(ARTIFACTS / "scene/trace.npz")
    path_d = scene.label(trace["q"])
    test_indices = np.random.default_rng(77).choice(len(initial["test_q"]), 100000, replace=False)
    datasets = {"global_test": (initial["test_q"][test_indices], initial["test_d"][test_indices]),
                "global_validation": (initial["validation_q"], initial["validation_d"]),
                "task_validation": (pools["validation"]["q"], pools["validation"]["d"]),
                "task_test": (pools["test"]["q"], pools["test"]["d"]),
                "original_path": (trace["q"], path_d),
                "original_stage6": (trace["q"][trace["stage"] == 6], path_d[trace["stage"] == 6])}
    np.savez_compressed(args.output / "evaluation_inputs.npz", **{f"{key}_{part}": values[i]
                        for key, values in datasets.items() for i, part in enumerate(("q", "d"))})
    np.savez_compressed(args.output / "replay.npz", q=old_q, d=old_d)
    for method, q, d in (("baseline", None, None), ("control", old_q, old_d), ("mixed", mixed_q, mixed_d)):
        torch.manual_seed(config.seed)
        model, checkpoint = load_checkpoint(checkpoint_path, scene)
        model = model.to(args.device)
        began = time.perf_counter()
        if q is not None:
            losses = fit_block(model, q, d, config, np.random.default_rng(config.seed))
            checkpoint.update(model=model.cpu().state_dict(), config=asdict(config), method=method,
                              qualified=False, baseline_sha256=baseline_hash, joint_names=list(JOINTS), round=0)
            torch.save(checkpoint, args.output / f"{method}.pt")
            model.to(args.device)
            report["methods"][method] = {"first_loss": losses[0], "last_loss": losses[-1],
                                           "optimizer_steps": args.epochs * ((total + config.batch_size - 1) // config.batch_size)}
        else:
            report["methods"][method] = {}
        report["methods"][method]["training_s"] = time.perf_counter() - began
        report["methods"][method]["evaluation"] = evaluate(model, datasets, args.output, method)
        write_json(args.output / "progress.json", report)
        print(f"{method}: {report['methods'][method]['evaluation']['task_test']}", flush=True)
    if file_hash(checkpoint_path) != baseline_hash:
        raise RuntimeError("실험 중 원본 체크포인트가 바뀌었습니다.")
    report["baseline_unchanged"] = True
    report["elapsed_s"] = time.perf_counter() - started
    write_json(args.output / "results.json", report)


def main():
    """별도 출력 폴더에만 기록하는 작은 삽입 자료 실험을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ARTIFACTS / "full")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "insertion_smoke")
    parser.add_argument("--task-count", type=int, default=30000)
    parser.add_argument("--candidates-per-family", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    run(args)


if __name__ == "__main__":
    main()
