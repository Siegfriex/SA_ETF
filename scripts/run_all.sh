#!/usr/bin/env bash
# 전체 파이프라인 (C-0021): validate → build → run → profile → inject. 각 단계 exit code 출력, 실패 시 중단.
# profile 은 notebook 실행 결과(profiles json)가 필요하므로 run 뒤에 둔다. raw 는 immutable (collect 는 별도 수동).
# env: ETF_SHARED_DATA (기본 .../ETF_EDA_SCAFFOLD/data), ETF_RAW_DIR, PY (기본 수업 저장소 .venv)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python}
for step in validate_raw build_notebooks run_notebooks profile_universe inject_cards; do
  "$PY" "scripts/$step.py"; rc=$?
  echo "STEP $step exit=$rc"
  [ $rc -eq 0 ] || exit $rc
done
