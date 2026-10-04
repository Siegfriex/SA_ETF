#!/usr/bin/env bash
# Phase 00.5 final pipeline (C2-B1): build 20 final notebooks → execute (병렬) → universe (A) → robustness (D).
# 단계별 exit code 출력, 실패 시 중단. env: PY (기본 수업 저장소 .venv), ETF_RAW_DIR (raw 경로; clean clone 에서는 공유 data/raw 로 지정), JOBS (기본 6)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python}
JOBS=${JOBS:-6}
JUPYTER=${JUPYTER:-/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/jupyter}
export ETF_RAW_DIR=${ETF_RAW_DIR:-$(pwd)/data/raw}
export MPLBACKEND=${MPLBACKEND:-Agg}
step() { echo "STEP $1 exit=$2"; [ "$2" -eq 0 ] || exit "$2"; }

"$PY" scripts/build_final_notebooks.py; step build_final_notebooks $?

ls notebooks/final_eda/*_final_eda.ipynb | xargs -P "$JOBS" -I{} "$JUPYTER" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=hongik_26_2 --ExecutePreprocessor.timeout=900 {} --log-level=WARN
step execute_final_notebooks $?
n=$(ls notebooks/final_eda/*_final_eda.ipynb | wc -l); echo "final notebooks: $n"; [ "$n" -eq 20 ] || step count_20 1

if [ -f analysis/scripts/universe_final.py ]; then "$PY" analysis/scripts/universe_final.py; step universe_final $?; else echo "SKIP universe_final.py (없음)"; fi
for nb in notebooks/universe/*.ipynb; do
  [ -f "$nb" ] || { echo "SKIP universe notebook (없음)"; break; }
  "$JUPYTER" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=hongik_26_2 \
    --ExecutePreprocessor.timeout=1800 "$nb" --log-level=WARN; step "universe_nb:$(basename "$nb")" $?
done
if [ -f analysis/robustness/robustness_final.py ]; then "$PY" analysis/robustness/robustness_final.py; step robustness_final $?; else echo "SKIP robustness_final.py (없음)"; fi
if [ -f analysis/robustness/d0122_window_loetf.py ]; then "$PY" analysis/robustness/d0122_window_loetf.py >/dev/null; step d0122_window_loetf $?; fi
echo "ALL DONE"
