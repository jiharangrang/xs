#!/usr/bin/env bash
# CUDA 소규모 학습과 동일 경로 비교·테스트를 실행하고 종료 코드를 보존한다.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e Learning/artifacts/smoke ]]; then
  echo "기존 smoke 결과가 있어 덮어쓰지 않습니다."
  exit 1
fi
mkdir -p Learning/artifacts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/Learning/artifacts/.mpl-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/Learning/artifacts/.compile-cache"
trap 'run_status=$?; printf "%s\n" "$run_status" > Learning/artifacts/smoke.exitcode; printf "소규모 검증 종료 코드: %s\n" "$run_status"' EXIT
.venv/bin/python -u Learning/train.py --device cuda --workers 8 --threads 4 --profile smoke --method both 2>&1 | tee Learning/artifacts/smoke.log
.venv/bin/python -u Learning/compare.py 2>&1 | tee -a Learning/artifacts/smoke.log
.venv/bin/python -m unittest discover -s Learning/tests -v 2>&1 | tee -a Learning/artifacts/smoke.log
