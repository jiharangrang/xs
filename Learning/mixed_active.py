"""무작위 초기화한 SE3NN을 전역·능동 경계·삽입 자료의 고정 비율로 학습한다.
이전 가중치나 능동 추출 자료를 재사용하지 않고 별도 결과 폴더에 모든 회차를 보존한다.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from active import boundary_samples
from common import ARTIFACTS, JOINTS, REPO, file_hash, write_json
from insertion_data import candidate_pool, distribution, select_training
from insertion_experiment import detailed_metrics, evaluate
from model import SE3NN
from parallel import ParallelLabeler
from scene import DistanceScene
from train import TrainConfig, fit_block, predict


COUNTS = {"uniform": 35000, "active": 35000, "insertion": 30000}


def size_settings(train_size):
    """10만·100만 실행에서 자료 비율과 경로 묶음 수를 함께 확대한다."""
    if train_size not in (100000, 1000000):
        raise ValueError("학습셋 크기는 100000 또는 1000000으로 지정하세요.")
    scale = train_size // 100000
    return {"counts": {name: count * scale for name, count in COUNTS.items()},
            "train_families": 120 * scale, "batch_size": 1000 * scale,
            "profile": "mixed_100k_scratch" if scale == 1 else "mixed_1m_scratch"}


def assemble_dataset(parts, counts=COUNTS):
    """세 자료 묶음의 크기를 검증하고 원본을 바꾸지 않은 채 학습 배열로 합친다."""
    if set(parts) != set(counts):
        raise ValueError("전역·능동·삽입 자료 묶음이 모두 필요합니다.")
    for name, count in counts.items():
        q, d = parts[name]
        if (q.shape != (count, len(JOINTS)) or d.shape != (count,)
                or not np.isfinite(q).all() or not np.isfinite(d).all()):
            raise ValueError(f"{name} 자료의 개수·형상·유한성이 올바르지 않습니다.")
    q = np.concatenate([parts[name][0] for name in counts]).astype(np.float32)
    d = np.concatenate([parts[name][1] for name in counts]).astype(np.float32)
    source = np.concatenate([np.full(count, index, dtype=np.uint8) for index, count in enumerate(counts.values())])
    return q, d, source


def uniform_data(scene, count, rng, labeler):
    """관절 제한 안에서 새 균등 자세를 뽑고 같은 메시 장면의 정답을 계산한다."""
    q = rng.uniform(scene.limits[:, 0], scene.limits[:, 1], (count, len(JOINTS))).astype(np.float32)
    return q, labeler.label(q, progress=True)


def sample_active(model, scene, task_q, rng, device, *, count=35000, steps=1000):
    """전역 및 삽입 시작점에서 현재 모델의 영거리 경계를 각각 탐색한다."""
    global_count = count // 2
    global_q, global_info = boundary_samples(model, scene.limits, global_count, rng,
        sigma_e=.1, proposal_std=.05, steps=steps, device=device)
    indices = rng.choice(len(task_q), count - global_count, replace=False)
    local_q, local_info = boundary_samples(model, scene.limits, count - global_count, rng,
        sigma_e=.005, proposal_std=.005, steps=steps, device=device, initial_q=task_q[indices])
    return np.concatenate((global_q, local_q)), {"global": global_info, "insertion_seeded": local_info,
                                               "global_count": len(global_q), "insertion_seeded_count": len(local_q)}


def save_checkpoint(model, path, scene, config, iteration, seed):
    """GPU 모델의 위치를 바꾸지 않고 회차별 가중치와 초기화 근거를 저장한다."""
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"model": state, "scene_id": scene.scene_id, "joint_names": list(JOINTS),
                "config": asdict(config), "round": iteration, "method": "mixed_active_from_scratch",
                "initialization": "random", "initial_seed": seed, "pretrained_checkpoint": None,
                "qualified": False}, path)


def run(args):
    """새 자료와 새 가중치로 초기 학습 후 능동 자료만 교체하며 정해진 횟수만 학습한다."""
    if args.output.exists():
        raise FileExistsError(f"기존 실행 결과를 덮어쓰지 않습니다: {args.output}")
    if args.epochs < 1 or args.active_rounds < 1 or not 1 <= args.workers <= 8:
        raise ValueError("학습 횟수는 양수이고 작업 프로세스는 1~8개여야 합니다.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    started = time.perf_counter()
    scene = DistanceScene()
    settings = size_settings(args.train_size)
    counts = settings["counts"]
    config = TrainConfig(train_size=args.train_size, validation_size=10000, test_size=100000,
                         epochs=args.epochs, training_rounds=args.active_rounds + 1, replace_size=counts["active"],
                         batch_size=settings["batch_size"] if args.batch_size is None else args.batch_size,
                         mcmc_steps=1000, seed=args.seed, device=args.device,
                         label_workers=args.workers, compile_model=args.device == "cuda", profile=settings["profile"])
    config.validate()
    args.output.mkdir(parents=True)
    report = {"scene_id": scene.scene_id, "config": asdict(config), "counts": counts,
              "initialization": "random", "pretrained_checkpoint": None, "prior_training_data_used": False,
              "bootstrap": "첫 학습의 능동 자료 칸은 새 균등 자료로 채움; 이후 현재 모델의 MCMC 결과로 전량 교체",
              "selection": "미리 정한 마지막 회차를 사용; 시험 자료는 학습·추출·선택에 사용하지 않음",
              "learning_rates": {"bootstrap": .001, "active_rounds": .0001},
              "active_sampling": {"global": {"count": counts["active"] // 2, "sigma_e_m": .1, "proposal_std_rad": .05},
                                  "insertion_seeded": {"count": counts["active"] // 2, "sigma_e_m": .005, "proposal_std_rad": .005},
                                  "steps": 1000, "u_min": .8},
              "adaptations": ["XS 작업용 35:35:30 비율 유지", "능동 자료 칸만 교체", "절반의 MCMC를 삽입 학습 자세에서 시작",
                              "삽입 시작 MCMC는 작은 거리·관절 척도 사용", "초기 학습 후 낮은 학습률 사용"],
              "data": {}, "rounds": [], "status": "generating_data"}
    report["execution"] = {"torch_version": str(torch.__version__), "device": args.device,
                           "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                           "label_workers": args.workers, "threads": torch.get_num_threads()}
    report["scaling"] = {"train_families": settings["train_families"], "candidates_per_family": 1200,
                         "evaluation_sizes_unchanged": True,
                         "default_batch_policy": "자료 크기와 배치를 함께 확대하여 epoch당 갱신 횟수 유지"}
    report["source_sha256"] = {name: file_hash(REPO / name) for name in (
        "Learning/mixed_active.py", "Learning/active.py", "Learning/train.py", "Learning/insertion_data.py",
        "Learning/insertion_experiment.py", "Learning/model.py", "Learning/scene.py", "kinematics/ik.py")}
    write_json(args.output / "config.json", report)
    write_json(args.output / "progress.json", report)
    pools = {}
    with ParallelLabeler(scene, args.workers) as labeler:
        for offset, (split, families, per_family) in enumerate((
                ("train", settings["train_families"], 1200), ("validation", 20, 160), ("test", 32, 200))):
            print(f"새 삽입 {split} 후보 생성: {families * per_family}개", flush=True)
            pool, info = candidate_pool(split, families, per_family, args.seed + 10000 + 200000 * offset,
                                        ARTIFACTS / "scene/trace.npz", args.workers, progress=True)
            pool["d"] = labeler.label(pool["q"], progress=True)
            info["candidate_distribution"] = distribution(pool)
            np.savez_compressed(args.output / f"{split}_candidates.npz", **pool)
            if split == "train":
                pool = select_training(pool, counts["insertion"], np.random.default_rng(args.seed + 1))
            pools[split] = pool
            info["selected_distribution"] = distribution(pool)
            report["data"][split] = info
            np.savez_compressed(args.output / f"task_{split}.npz", **pool)
            write_json(args.output / "progress.json", report)
            print(f"새 삽입 {split}: {info['selected_distribution']}", flush=True)
        parts = {"insertion": (pools["train"]["q"], pools["train"]["d"])}
        for offset, name in enumerate(("uniform", "active")):
            print(f"새 {name} 자료 정답 생성: {counts[name]}개", flush=True)
            parts[name] = uniform_data(scene, counts[name], np.random.default_rng(args.seed + 10 + offset), labeler)
        datasets = {}
        for offset, (name, count) in enumerate((("global_validation", 10000), ("global_test", 100000))):
            print(f"새 {name} 자료 정답 생성: {count}개", flush=True)
            datasets[name] = uniform_data(scene, count, np.random.default_rng(args.seed + 20 + offset), labeler)
        for split in ("validation", "test"):
            datasets[f"task_{split}"] = (pools[split]["q"], pools[split]["d"])
        with np.load(ARTIFACTS / "scene/trace.npz") as trace:
            q_path, stages = trace["q"].copy(), trace["stage"].copy()
        path_d = labeler.label(q_path)
        datasets["original_path"] = (q_path, path_d)
        datasets["original_stage6"] = (q_path[stages == 6], path_d[stages == 6])
        np.savez_compressed(args.output / "evaluation_inputs.npz", **{f"{key}_{part}": values[i]
                            for key, values in datasets.items() for i, part in enumerate(("q", "d"))})
        report["data_generation_s"] = time.perf_counter() - started
        torch.manual_seed(args.seed)
        model = SE3NN(scene.model).to(args.device)
        save_checkpoint(model, args.output / "initial_random.pt", scene, config, -1, args.seed)
        report["initial_checkpoint_sha256"] = file_hash(args.output / "initial_random.pt")
        network = torch.compile(model.network, mode="reduce-overhead") if config.compile_model else model.network
        sampler = torch.compile(model, mode="reduce-overhead") if config.compile_model else model
        train_rng = np.random.default_rng(args.seed + 30)
        active_rng = np.random.default_rng(args.seed + 31)
        for iteration in range(args.active_rounds + 1):
            entry = {"round": iteration, "bootstrap": iteration == 0, "counts": counts}
            if iteration:
                report["status"] = f"sampling_round_{iteration}"
                write_json(args.output / "progress.json", report)
                began = time.perf_counter()
                new_q, info = sample_active(sampler, scene, parts["insertion"][0], active_rng, args.device,
                                            count=counts["active"], steps=config.mcmc_steps)
                entry["sampling_s"] = time.perf_counter() - began
                began = time.perf_counter()
                new_d = labeler.label(new_q, progress=True)
                entry["labeling_s"] = time.perf_counter() - began
                parts["active"] = new_q, new_d
                entry["mcmc"] = info
                entry["active_near_5mm_fraction"] = float((np.abs(new_d) <= .005).mean())
            q, d, source = assemble_dataset(parts, counts)
            np.savez_compressed(args.output / f"dataset_{iteration:02d}.npz", q=q, d=d, source=source)
            round_config = replace(config, learning_rate=.001 if iteration == 0 else .0001)
            entry["learning_rate"] = round_config.learning_rate
            report["status"] = f"training_round_{iteration}"
            write_json(args.output / "progress.json", report)
            print(f"학습 회차 {iteration}/{args.active_rounds}, {len(q):,}개, 초기 준비={iteration == 0}", flush=True)
            began = time.perf_counter()
            losses = fit_block(model, q, d, round_config, train_rng, network, progress_every=100)
            entry["training_s"] = time.perf_counter() - began
            entry["first_train_mse"], entry["last_train_mse"] = losses[0], losses[-1]
            entry["optimizer_steps"] = config.epochs * ((len(q) + config.batch_size - 1) // config.batch_size)
            np.savez_compressed(args.output / f"loss_{iteration:02d}.npz", mse=np.asarray(losses))
            checkpoint = args.output / f"round_{iteration:02d}.pt"
            save_checkpoint(model, checkpoint, scene, round_config, iteration, args.seed)
            entry["checkpoint_sha256"] = file_hash(checkpoint)
            entry["validation"] = {name: detailed_metrics(predict(model, datasets[name][0]), datasets[name][1])
                                   for name in ("global_validation", "task_validation")}
            report["rounds"].append(entry)
            write_json(args.output / "progress.json", report)
            summary = {name: values["mae_m"] * 1000 for name, values in entry["validation"].items()}
            print(f"회차 {iteration} 검증 MAE(mm): {summary}", flush=True)
        shutil.copyfile(checkpoint, args.output / "model.pt")
        report["status"] = "evaluating"
        write_json(args.output / "progress.json", report)
        report["evaluation"] = evaluate(model, datasets, args.output, "final")
        report["elapsed_s"] = time.perf_counter() - started
        report["status"] = "complete"
        write_json(args.output / "results.json", report)
        write_json(args.output / "progress.json", report)
        print(f"처음부터 학습 완료: {report['elapsed_s']:.1f}초", flush=True)


def main():
    """선택한 크기의 학습셋을 고정하고 초기 학습 이후 지정 횟수만큼 능동 갱신한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-size", type=int, choices=(100000, 1000000), default=100000)
    parser.add_argument("--batch-size", type=int, help="기본 배치는 10만 자료에서 1000, 100만 자료에서 10000")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--active-rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output is None:
        args.output = ARTIFACTS / size_settings(args.train_size)["profile"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    run(args)


if __name__ == "__main__":
    main()
