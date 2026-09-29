#!/usr/bin/env bash
# 원본 가중치를 보존하고 삽입 자료 대조 실험과 검증을 별도 폴더에 기록한다.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e Learning/artifacts/insertion_smoke ]]; then
  echo "기존 삽입 실험이 있어 덮어쓰지 않습니다."
  exit 1
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
trap 'run_status=$?; printf "%s\n" "$run_status" > Learning/artifacts/insertion_smoke.exitcode; printf "삽입 실험 종료 코드: %s\n" "$run_status"' EXIT
.venv/bin/python -m unittest discover -s Learning/tests -v 2>&1 | tee Learning/artifacts/insertion_smoke.log
.venv/bin/python -u Learning/insertion_experiment.py 2>&1 | tee -a Learning/artifacts/insertion_smoke.log
