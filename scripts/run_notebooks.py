"""B3 runner — notebook 을 ETF 별 독립 nbconvert 프로세스(fresh kernel)로 실행하고 RUN_MANIFEST 를 쓴다.

상태:
- FAIL_DATA  : raw 없음 / 행 부족 (실행 전 판정)
- FAIL_CODE  : 실행 중 예외 (nbconvert 비정상 종료). 출력 사본은 *_failed.ipynb (gitignored)
- PASS_WITH_LIMITATION : 실행 성공 + profile.run_limitations 에 universe 공통 한계 외 항목 존재
- PASS       : 실행 성공, ETF 고유 한계 없음
universe 공통 한계(전 ETF 동일)는 UNIVERSE_LIMITATIONS 로 manifest 최상단에 한 번만 기록한다.

usage: python scripts/run_notebooks.py [ticker ...]   (인자 없으면 universe 전체)
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARED_DATA = Path(os.environ.get("ETF_SHARED_DATA", "/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD/data"))
RAW_DIR = Path(os.environ.get("ETF_RAW_DIR", SHARED_DATA / "raw"))
VENV = Path("/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin")
NB_DIR = ROOT / "notebooks" / "00_raw_eda"
PROFILE_DIR = ROOT / "data" / "interim" / "profiles"
KERNEL = "hongik_26_2"
TIMEOUT = 600
MIN_ROWS = 250
WORKERS = 2  # 4 에서 ZMQ 포트 충돌 1회 관측
KST = timezone(timedelta(hours=9))
UNIVERSE_LIMITATIONS = {"adjusted_close_estimated"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pkg_versions():
    out = subprocess.run([str(VENV / "python"), "-c",
                          "import importlib.metadata as m;"
                          "print({k: m.version(k) for k in ['pandas','numpy','matplotlib','scipy','statsmodels',"
                          "'scikit-learn','finance-datareader','pykrx','nbconvert','nbformat','ipykernel']})"],
                         capture_output=True, text=True)
    try:
        return eval(out.stdout.strip())  # dict literal from our own subprocess
    except Exception:
        return {"error": out.stderr[-300:]}


def run_one(row):
    t, slot = row["ticker"], int(row["slot"])
    nb = NB_DIR / f"00_{slot:02d}_{t}_raw_eda.ipynb"
    rec = {"slot": slot, "ticker": t, "name": row["name"], "notebook": str(nb.relative_to(ROOT))}
    raw = RAW_DIR / f"{t}.csv"
    if not raw.exists():
        return rec | {"status": "FAIL_DATA", "reason": f"raw missing {raw}"}
    rec["raw_sha256"] = sha256(raw)
    n = sum(1 for _ in open(raw, encoding="utf-8")) - 1
    if n < MIN_ROWS:
        return rec | {"status": "FAIL_DATA", "reason": f"rows={n} < {MIN_ROWS}"}
    if not nb.exists():
        return rec | {"status": "FAIL_CODE", "reason": "notebook not generated"}
    t0 = time.time()
    tmp = nb.with_name(nb.stem + ".exec.ipynb")
    p = subprocess.run([str(VENV / "jupyter"), "nbconvert", "--to", "notebook", "--execute",
                        f"--ExecutePreprocessor.kernel_name={KERNEL}", f"--ExecutePreprocessor.timeout={TIMEOUT}",
                        "--output", tmp.name, str(nb)],
                       cwd=nb.parent, capture_output=True, text=True)
    rec["seconds"] = round(time.time() - t0, 1)
    if p.returncode != 0:
        if tmp.exists():
            tmp.rename(nb.with_name(nb.stem + "_failed.ipynb"))
        err = [l for l in p.stderr.splitlines() if "Error" in l or "Exception" in l]
        return rec | {"status": "FAIL_CODE", "reason": (err[-1] if err else p.stderr[-300:])[:400]}
    tmp.replace(nb)  # 실행 결과(출력 포함)를 정본 notebook 으로
    rec["notebook_bytes"] = nb.stat().st_size
    prof = PROFILE_DIR / f"{t}.json"
    lims = []
    if prof.exists():
        lims = json.loads(prof.read_text(encoding="utf-8")).get("run_limitations", []) or []
    else:
        lims = ["profile_missing"]
    own = sorted(set(lims) - UNIVERSE_LIMITATIONS)
    rec["limitations"] = own
    rec["status"] = "PASS_WITH_LIMITATION" if own else "PASS"
    return rec


def main(only):
    u = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False)
    if only:
        u = u[u["ticker"].isin(only)]
    started = datetime.now(KST)

    def safe(row):
        try:
            return run_one(row)
        except Exception as e:  # runner 자체 오류도 ETF 단위로 격리
            return {"slot": int(row["slot"]), "ticker": row["ticker"], "status": "FAIL_CODE", "reason": f"runner: {e}"}

    recs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:  # 각 작업은 별도 nbconvert 프로세스 = fresh kernel
        for r in ex.map(safe, [row for _, row in u.iterrows()]):
            recs.append(r)
            print(f"{r['ticker']} {r['status']} {r.get('seconds', '')}s {r.get('reason', '') or ','.join(r.get('limitations', []))}",
                  flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest_path = ROOT / "reports" / "RUN_MANIFEST.json"
    prev = json.loads(manifest_path.read_text(encoding="utf-8")) if (only and manifest_path.exists()) else {}
    merged = {r["ticker"]: r for r in prev.get("runs", [])}
    merged.update({r["ticker"]: r for r in recs})
    runs = sorted(merged.values(), key=lambda r: r["slot"])
    counts = pd.Series([r["status"] for r in runs]).value_counts().to_dict()
    manifest = {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "started_at": started.isoformat(timespec="seconds"),
        "git_sha_at_run": sha,
        "universe_status": "PROVISIONAL",
        "kernel": KERNEL, "python": platform.python_version(),
        "packages": pkg_versions(),
        "raw_dir": str(RAW_DIR),
        "UNIVERSE_LIMITATIONS": {"adjusted_close_estimated": "FDR Close 분배 소급조정 추정(069500/KS200 1.155 vs TR 1.163), O/H/L 조정 일관성 미보장 — 전 ETF 공통 (C-0012 DEC-3)"},
        "counts": counts,
        "runs": runs,
    }
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("counts", counts, "→", manifest_path)


if __name__ == "__main__":
    main(sys.argv[1:])
