#!/usr/bin/env bash
# 새 가중치와 새 자료로 혼합 능동 학습을 실행하고 종료 상태와 회차별 결과를 보존한다.
set -euo pipefail
cd "$(dirname "$0")/.."
dataset_size=${1:-100000}
case "$dataset_size" in
  100000) run_name=mixed_100k_scratch ;;
  1000000) run_name=mixed_1m_scratch ;;
  *) echo "학습셋 크기는 100000 또는 1000000으로 지정하세요."; exit 2 ;;
esac
if [[ -e "Learning/artifacts/$run_name" ]]; then
  echo "기존 처음부터 학습한 결과가 있어 덮어쓰지 않습니다."
  exit 1
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/Learning/artifacts/.mpl-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/Learning/artifacts/.compile-cache"
trap 'run_status=$?; printf "%s\n" "$run_status" > "Learning/artifacts/$run_name.exitcode"; printf "혼합 능동 학습 종료 코드: %s\n" "$run_status"' EXIT
.venv/bin/python -m unittest discover -s Learning/tests -v 2>&1 | tee "Learning/artifacts/$run_name.log"
.venv/bin/python -u Learning/mixed_active.py --train-size "$dataset_size" 2>&1 | tee -a "Learning/artifacts/$run_name.log"
.venv/bin/python -u Learning/compare.py --checkpoint "Learning/artifacts/$run_name/model.pt" --output "Learning/artifacts/$run_name/comparison" 2>&1 | tee -a "Learning/artifacts/$run_name.log"
