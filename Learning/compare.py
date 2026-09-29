"""동일한 6단계 관절 경로에서 메시와 학습 모델의 순차 거리 질의를 비교한다.
순기구학을 포함한 실제 계산 시간과 거리 오차를 함께 기록하며 모터 속도 향상으로 해석하지 않는다.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from common import ARTIFACTS, file_hash, write_json
from model import load_checkpoint
from scene import DistanceScene
from train import metrics


def benchmark(scene, model, q, *, repeats=3):
    """캐시 준비 뒤 같은 자세를 한 개씩 조회하고 두 방식의 실행 순서를 번갈아 측정한다."""
    if repeats < 1 or len(q) == 0:
        raise ValueError("비어 있지 않은 경로와 양의 반복 횟수가 필요합니다.")
    model.eval()
    results = {}
    with torch.inference_mode():
        for sample in q[:8]:
            scene.distance(sample)
            model(torch.tensor(sample[None], dtype=torch.float32))
        for repeat in range(repeats):
            order = ("mesh", "learned") if repeat % 2 == 0 else ("learned", "mesh")
            for name in order:
                distances, milliseconds = [], []
                start_all = time.perf_counter()
                for sample in q:
                    started = time.perf_counter_ns()
                    if name == "mesh":
                        distance = scene.distance(sample)
                    else:
                        distance = float(model(torch.tensor(sample[None], dtype=torch.float32))[0])
                    elapsed = time.perf_counter_ns() - started
                    milliseconds.append(elapsed / 1e6)
                    distances.append(distance)
                entry = results.setdefault(name, {"milliseconds": [], "total_s": [], "distances": distances})
                entry["milliseconds"].append(milliseconds)
                entry["total_s"].append(time.perf_counter() - start_all)
    return results


def save_plot(path, stages, results):
    """각 단계의 실제 계산 시간과 전체 경로의 거리 오차를 정적 그래프로 저장한다."""
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    positions = np.arange(1, 7)
    for name, shift in (("mesh", -.18), ("learned", .18)):
        latencies = np.asarray(results[name]["milliseconds"]).mean(axis=0)
        sums = [latencies[stages == stage].sum() for stage in positions]
        axes[0].bar(positions + shift, sums, width=.36, label=name)
        axes[1].plot(np.asarray(results[name]["distances"]) * 1000, label=name)
    axes[0].set(xlabel="Ideal stage", ylabel="Total query time (ms)", xticks=positions)
    axes[1].set(xlabel="Same replay sample", ylabel="Signed distance (mm)")
    axes[1].axhline(0, color="black", linewidth=.8)
    for axis in axes:
        axis.legend()
    figure.suptitle("Sequential queries including FK; smoke model is not accuracy-qualified")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def compare(scene_path, checkpoint, output, *, repeats=3, stride=1):
    """완료된 경로의 모든 단계에 같은 조회를 넣고 시간·오류를 함께 저장한다."""
    if stride < 1:
        raise ValueError("stride는 양수여야 합니다.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"기존 비교 결과를 덮어쓰지 않습니다: {output}")
    scene = DistanceScene(scene_path)
    model, saved = load_checkpoint(checkpoint, scene)
    with np.load(Path(scene_path) / "trace.npz", allow_pickle=False) as trace:
        q, stages = trace["q"][::stride], trace["stage"][::stride]
    if set(stages) != set(range(1, 7)):
        raise ValueError("비교 경로에 1–6단계가 모두 필요합니다.")
    measured = benchmark(scene, model, q, repeats=repeats)
    report = {"scene_id": scene.scene_id, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
              "profile": saved["config"]["profile"], "qualified": saved["qualified"],
              "threads": torch.get_num_threads(), "samples": len(q), "repeats": repeats,
              "trace_sha256": file_hash(Path(scene_path) / "trace.npz"),
              "interpretation": "고정된 이론 경로에서 반복 거리 질의의 실제 처리시간. 모터·센서·GUI·재계획 시간 제외.",
              "scope": scene.config["scope"], "geometry": "CAD convex decomposition, not exact original triangle-soup distance",
              "error": metrics(measured["learned"]["distances"], measured["mesh"]["distances"]), "timing": {}, "stages": {}}
    for name in ("mesh", "learned"):
        ms = np.asarray(measured[name]["milliseconds"])
        report["timing"][name] = {"mean_ms": float(ms.mean()), "p50_ms": float(np.median(ms)),
                                   "p95_ms": float(np.percentile(ms, 95)),
                                   "path_processing_s": float(np.mean(measured[name]["total_s"]))}
    report["query_speedup"] = report["timing"]["mesh"]["mean_ms"] / report["timing"]["learned"]["mean_ms"]
    for stage in range(1, 7):
        selected = stages == stage
        report["stages"][str(stage)] = {"samples": int(selected.sum()),
            "error": metrics(np.asarray(measured["learned"]["distances"])[selected],
                             np.asarray(measured["mesh"]["distances"])[selected])}
    output.mkdir(parents=True)
    write_json(output / "comparison.json", report)
    np.savez_compressed(output / "comparison.npz", q=q, stage=stages,
                        true_distance=measured["mesh"]["distances"], predicted_distance=measured["learned"]["distances"],
                        mesh_ms=measured["mesh"]["milliseconds"], learned_ms=measured["learned"]["milliseconds"])
    save_plot(output / "comparison.png", stages, measured)
    print(json.dumps({"timing": report["timing"], "speedup": report["query_speedup"],
                      "error": report["error"], "qualified": report["qualified"]}, indent=2), flush=True)
    return report


def main():
    """기존 경로에서 실제 계산 시간을 측정하며 선택적으로 경로를 성기게 뽑는다."""
    parser = argparse.ArgumentParser(description="6단계 메시/SE3NN 거리 조회 비교")
    parser.add_argument("--scene", type=Path, default=ARTIFACTS / "scene")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "smoke/active.pt")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("스레드는 1~8개로 지정하세요.")
    torch.set_num_threads(args.threads)
    compare(args.scene, args.checkpoint, args.output, repeats=args.repeats, stride=args.stride)


if __name__ == "__main__":
    main()
