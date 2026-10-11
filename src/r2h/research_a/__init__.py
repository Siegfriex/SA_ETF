"""R2H Track A — cross-asset EDA (KOSPI200 vs 채권·해외주식·코스닥·원자재·FX).

기존 src/eda_utils.py · src/final_figs.py 는 READ-ONLY. A 는 이 패키지에서만 로직을 추가한다.
src/r2h/ 는 Track B 와 공유되는 namespace 경로이므로 src/r2h/__init__.py 를 두지 않는다.
"""
from .contract import Contract, input_data_version, load_contract, make_run_id  # noqa: F401
from .data import etf_manifest, load_close_panel, load_interim, log_returns, returns, verify_raw_sha  # noqa: F401
