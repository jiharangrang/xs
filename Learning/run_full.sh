#!/usr/bin/env bash
# 논문 본문의 학습량으로 액티브 SE3NN을 학습하고 동일 경로의 거리 조회를 비교한다.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e Learning/artifacts/full || -e Learning/artifacts/comparison_full ]]; then
  echo "기존 본학습 또는 비교 결과가 있어 덮어쓰지 않습니다."
  exit 1
fi
mkdir -p Learning/artifacts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/Learning/artifacts/.mpl-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/Learning/artifacts/.compile-cache"
trap 'run_status=$?; printf "%s\n" "$run_status" > Learning/artifacts/full.exitcode; printf "본학습 작업 종료 코드: %s\n" "$run_status"' EXIT
date -Is | tee Learning/artifacts/full.log
.venv/bin/python -u Learning/train.py --device cuda --workers 8 --threads 4 --profile full --method active --output Learning/artifacts/full 2>&1 | tee -a Learning/artifacts/full.log
.venv/bin/python -u Learning/compare.py --checkpoint Learning/artifacts/full/active.pt --output Learning/artifacts/comparison_full 2>&1 | tee -a Learning/artifacts/full.log
