"""고정 크기 데이터로 SE3NN을 학습하고 논문 방식의 경계 표본으로 갱신한다.
기본 실행은 소규모 검증이며 대량 학습은 full 프로필을 명시해야 한다.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata

from active import boundary_samples, replace_random
from common import ARTIFACTS, JOINTS, write_json
from model import SE3NN
from parallel import ParallelLabeler
from scene import DistanceScene


@dataclass(frozen=True)
class TrainConfig:
    """표본·학습·경계 추출 예산과 논문에서 가져온 설정을 보관한다."""

    train_size: int = 2048
    validation_size: int = 512
    test_size: int = 512
    epochs: int = 120
    training_rounds: int = 3
    replace_size: int = 256
    batch_size: int = 256
    mcmc_steps: int = 100
    sigma_e: float = .1
    proposal_std: float = .05
    u_min: float = .8
    learning_rate: float = .001
    seed: int = 7
    near_m: float = .1
    profile: str = "smoke"
    device: str = "cpu"
    label_workers: int = 1
    compile_model: bool = False

    def validate(self):
        """무효한 예산과 경계 설정을 학습 전에 거부한다."""
        for key in ("train_size", "validation_size", "test_size", "epochs", "training_rounds", "replace_size", "batch_size", "mcmc_steps"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key}는 양의 정수여야 합니다.")
        if self.replace_size > self.train_size or not 0 <= self.u_min < 1:
            raise ValueError("반복 횟수 또는 데이터 교체 크기가 올바르지 않습니다.")
        for value in (self.sigma_e, self.proposal_std, self.learning_rate, self.near_m):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("학습 척도는 유한한 양수여야 합니다.")
        if self.device not in {"cpu", "cuda"} or not 1 <= self.label_workers <= 8:
            raise ValueError("장치 또는 라벨 계산 프로세스 수가 올바르지 않습니다.")
        if self.compile_model and self.device != "cuda":
            raise ValueError("컴파일 최적화는 CUDA 실행에만 적용합니다.")


def profile_config(name):
    """작은 실행과 논문 §5.3의 총 학습 횟수에 맞춘 명시적 대량 실행을 구분한다.

    알고리즘 1의 양끝 포함 인덱스와 §5.3의 총 횟수 설명이 다르므로 후자를 따른다.
    마지막 학습 뒤 사용하지 않을 자료 교체는 생략한다.
    """
    if name == "smoke":
        return TrainConfig()
    if name == "full":
        return TrainConfig(train_size=1_000_000, validation_size=10_000, test_size=1_000_000,
                           epochs=1000, training_rounds=20, replace_size=100_000,
                           batch_size=10_000, mcmc_steps=1000, profile="full")
    raise ValueError("지원하지 않는 학습 프로필입니다.")


def collision_auroc(predicted, truth):
    r"""충돌을 양성으로 두고 같은 점수의 순위를 평균하여 AUROC를 계산한다.

    $$\operatorname{AUROC}=\frac{\sum_{i:y_i=1}r_i-n_+(n_++1)/2}{n_+n_-}$$

    점수는 예측 거리의 음수이고 r은 오름차순 평균 순위다. 한 클래스만 있으면 정의하지 않는다.
    """
    collision = truth <= 0
    positive = int(collision.sum())
    negative = len(truth) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(-predicted, method="average")
    # 양성 표본의 순위합에서 양성끼리의 순위를 제외: $$U=\sum_{i:y_i=1}r_i-n_+(n_++1)/2$$
    statistic = ranks[collision].sum() - positive * (positive + 1) / 2
    # 모든 양성·음성 표본 쌍 중 올바른 순서의 비율: $$A=U/(n_+n_-)$$
    return float(statistic / (positive * negative))


def metrics(predicted, truth, near_m=.1):
    r"""거리 오차와 충돌을 안전하다고 놓치는 비율을 별도로 계산한다.

    $$MAE=\frac1N\sum_i|\hat d_i-d_i|,\quad FNR=\frac{\#\{d_i\leq0,\hat d_i>0\}}{\#\{d_i\leq0\}}$$

    기본 경계 범위는 논문 §5.1의 ±0.1 m이며 XS의 ±5 mm 평가는 별도로 유지한다.
    """
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if (predicted.ndim != 1 or predicted.shape != truth.shape or len(truth) == 0
            or not np.isfinite(predicted).all() or not np.isfinite(truth).all()):
        raise ValueError("거리 예측과 정답의 크기·유한성을 확인하세요.")
    # 거리 예측 오차: $$e=\hat d-d$$
    error = predicted - truth
    collision = truth <= 0
    missed = collision & (predicted > 0)
    near = np.abs(truth) <= near_m
    tight = np.abs(truth) <= .005
    correct = (predicted <= 0) == collision
    return {"count": len(truth), "mae_m": float(np.abs(error).mean()),
            "mse_m2": float(np.square(error).mean()),
            "collision_count": int(collision.sum()), "missed_collision_count": int(missed.sum()),
            "false_safe_rate": float(missed.sum() / collision.sum()) if collision.any() else None,
            "accuracy": float(np.mean(correct)), "auroc": collision_auroc(predicted, truth),
            "near_m": near_m, "near_count": int(near.sum()),
            "near_mae_m": float(np.abs(error[near]).mean()) if near.any() else None,
            "near_accuracy": float(correct[near].mean()) if near.any() else None,
            "near_auroc": collision_auroc(predicted[near], truth[near]),
            "tight_m": .005, "tight_count": int(tight.sum()),
            "tight_mae_m": float(np.abs(error[tight]).mean()) if tight.any() else None,
            "tight_accuracy": float(correct[tight].mean()) if tight.any() else None}


def predict(model, q, chunk=4096):
    """모델 장치에서 묶음 추론하고 결과만 CPU 배열로 반환한다."""
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        return np.concatenate([model(torch.as_tensor(q[i:i + chunk], dtype=torch.float32, device=device)).cpu().numpy()
                               for i in range(0, len(q), chunk)])


def fit_block(model, q, labels, config, rng, network=None, *, progress_every=0):
    r"""고정 데이터의 링크 표현을 캐시하고 논문의 거리 MSE를 Adam으로 줄인다.

    $$L(\theta)=\frac1N\sum_i(\hat d_\theta(q_i)-d_i)^2$$

    링크 표현은 가중치에 의존하지 않으므로 캐시해도 동일한 학습 목적함수다.
    """
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    with torch.no_grad():
        features = torch.cat([model.features(torch.from_numpy(q[i:i + 4096]).to(device)) for i in range(0, len(q), 4096)])
    targets = torch.from_numpy(labels).to(device)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=config.learning_rate, fused=True if cuda else None)
    network = model.network if network is None else network
    generator = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**31))) if cuda else None
    model.train()
    history = []
    for epoch in range(config.epochs):
        indices = torch.randperm(len(q), device=device, generator=generator) if cuda else rng.permutation(len(q))
        total = torch.zeros((), device=device) if cuda else 0.
        for begin in range(0, len(q), config.batch_size):
            batch = indices[begin:begin + config.batch_size]
            predicted = network(features[batch]).squeeze(-1)
            # 논문 식 (2)의 거리 평균제곱오차: $$L=\operatorname{mean}((\hat d-d)^2)$$
            loss = nn.functional.mse_loss(predicted, targets[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("학습 손실이 유한하지 않습니다.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += (loss.detach() if cuda else float(loss.detach())) * len(batch)
        history.append(float(total) / len(q))
        if progress_every and ((epoch + 1) % progress_every == 0 or epoch + 1 == config.epochs):
            print(f"학습 {epoch + 1}/{config.epochs}: MSE {history[-1]:.8g}", flush=True)
    return history


def generate_dataset(scene, config, labeler):
    """서로 다른 난수 흐름으로 학습·검증·최종 시험 자료를 독립 생성한다."""
    result = {}
    for index, (name, size) in enumerate((("train", config.train_size), ("validation", config.validation_size), ("test", config.test_size))):
        rng = np.random.default_rng(config.seed + index * 1009)
        q = rng.uniform(scene.limits[:, 0], scene.limits[:, 1], size=(size, len(JOINTS))).astype(np.float32)
        print(f"{name} 거리 정답 생성: {size}개", flush=True)
        result[name + "_q"] = q
        result[name + "_d"] = labeler.label(q, progress=True)
    return result


def run_training(scene, output, config, methods=("active", "uniform")):
    """지정한 장치와 거리 계산 풀을 준비하고 종료 시 모든 작업을 정리한다."""
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 드라이버와 PyTorch 설치를 확인하세요.")
    started = time.perf_counter()
    with ParallelLabeler(scene, config.label_workers) as labeler:
        report = _run_training(scene, output, config, methods, labeler)
    report["elapsed_s"] = time.perf_counter() - started
    if config.device == "cuda":
        report["execution"]["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    write_json(Path(output) / "training.json", report)
    return report


def _run_training(scene, output, config, methods, labeler):
    """같은 초기 자료·가중치·학습량으로 능동 학습과 고정 데이터 학습을 수행한다."""
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"기존 학습 결과를 덮어쓰지 않습니다: {output}")
    output.mkdir(parents=True)
    write_json(output / "config.json", {**asdict(config), "scene_id": scene.scene_id,
                                        "paper": "10.1017/S0263574723001790",
                                        "dataset_refreshes": config.training_rounds - 1,
                                        "total_epochs": config.training_rounds * config.epochs,
                                        "round_convention": "§5.3의 Nactive × Nepoch; 알고리즘 1의 양끝 포함 인덱스 대신 본문 총 학습량을 따름",
                                        "adaptations": ["XS 오른쪽 그리퍼-빔", "빔 기준 SE(3)",
                                                        "MuJoCo 볼록 분해 메시 거리", "sigma를 제안 표준편차로 해석",
                                                        "관절 범위 밖 MCMC 제안 거부", "수용 시 현재 밀도도 갱신",
                                                        "Adam과 학습률은 구현 선택; 각 학습 회차에 최적화 상태 초기화",
                                                        "독립 검증 분할", "CPU 또는 CUDA 학습", "고정 데이터 FK 입력 캐시",
                                                        "CUDA에서는 장치별 난수열로 MCMC와 배치 순서를 생성"]})
    began = time.perf_counter()
    dataset = generate_dataset(scene, config, labeler)
    dataset_s = time.perf_counter() - began
    np.savez_compressed(output / "dataset.npz", **dataset)
    report = {"profile": config.profile, "scene_id": scene.scene_id, "dataset_s": dataset_s, "methods": {},
              "execution": {"device": config.device, "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else "CPU",
                            "compile": config.compile_model, "label_workers": config.label_workers,
                            "torch_version": torch.__version__, "threads": torch.get_num_threads()}}
    for method in methods:
        if method not in {"active", "uniform"}:
            raise ValueError("학습 방식이 올바르지 않습니다.")
        torch.manual_seed(config.seed)
        model = SE3NN(scene.model).to(config.device)
        network = torch.compile(model.network, mode="reduce-overhead") if config.compile_model else model.network
        sampler_model = torch.compile(model, mode="reduce-overhead") if config.compile_model else model
        q, labels = dataset["train_q"].copy(), dataset["train_d"].copy()
        train_rng = np.random.default_rng(config.seed)
        active_rng = np.random.default_rng(config.seed + 7001)
        initial = metrics(predict(model, dataset["validation_q"]), dataset["validation_d"], config.near_m)
        rounds = []
        method_start = time.perf_counter()
        for iteration in range(config.training_rounds):
            start = time.perf_counter()
            losses = fit_block(model, q, labels, config, train_rng, network)
            validation = metrics(predict(model, dataset["validation_q"]), dataset["validation_d"], config.near_m)
            entry = {"round": iteration, "training_s": time.perf_counter() - start,
                     "first_train_mse": losses[0], "last_train_mse": losses[-1], "validation": validation,
                     "dataset_count": len(q), "near_fraction": float(np.mean(np.abs(labels) <= config.near_m))}
            rounds.append(entry)
            torch.save({"model": model.state_dict(), "scene_id": scene.scene_id, "joint_names": list(JOINTS),
                        "config": asdict(config), "method": method, "round": iteration,
                        "qualified": False}, output / f"{method}.pt")
            print(f"{method} {iteration + 1}/{config.training_rounds}: val MAE {validation['mae_m'] * 1000:.3f} mm", flush=True)
            write_json(output / f"{method}_progress.json", rounds)
            if method == "active" and iteration + 1 < config.training_rounds:
                start = time.perf_counter()
                new_q, diagnostics = boundary_samples(sampler_model, scene.limits, config.replace_size, active_rng,
                                                       sigma_e=config.sigma_e, proposal_std=config.proposal_std,
                                                       u_min=config.u_min, steps=config.mcmc_steps, device=config.device)
                sampling_s = time.perf_counter() - start
                label_start = time.perf_counter()
                new_labels = labeler.label(new_q, progress=True)
                entry["sampling_s"] = sampling_s
                entry["labeling_s"] = time.perf_counter() - label_start
                replace_random(q, labels, new_q, new_labels, active_rng)
                entry["update_s"] = time.perf_counter() - start
                entry["mcmc"] = diagnostics
                entry["replacement_count"] = len(new_q)
                entry["replacement_mean_abs_distance_m"] = float(np.abs(new_labels).mean())
                write_json(output / f"{method}_progress.json", rounds)
        test = metrics(predict(model, dataset["test_q"]), dataset["test_d"], config.near_m)
        report["methods"][method] = {"initial_validation": initial, "rounds": rounds, "test": test,
                                     "total_s": time.perf_counter() - method_start,
                                     "validation_mse_reduced": rounds[-1]["validation"]["mse_m2"] < initial["mse_m2"]}
        np.savez_compressed(output / f"{method}_final_dataset.npz", q=q, d=labels)
        write_json(output / "training.json", report)
    return report


def main():
    """기본 소규모 학습 또는 명시한 대량 프로필을 실행한다."""
    parser = argparse.ArgumentParser(description="SE3NN 거리 학습; 기본은 작은 smoke 실행")
    parser.add_argument("--scene", type=Path, default=ARTIFACTS / "scene")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "smoke")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--method", choices=("active", "uniform", "both"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="거리 정답 계산 프로세스 수, 최대 8")
    parser.add_argument("--no-compile", action="store_true", help="CUDA 컴파일 최적화를 끄고 실행")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 1 <= args.workers <= 8:
        parser.error("스레드와 작업 프로세스는 1~8개로 지정하세요.")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    config = replace(profile_config(args.profile), device=args.device, label_workers=args.workers,
                     compile_model=args.device == "cuda" and not args.no_compile)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    methods = ("active", "uniform") if args.method == "both" else (args.method,)
    run_training(DistanceScene(args.scene), args.output, config, methods)


if __name__ == "__main__":
    main()
