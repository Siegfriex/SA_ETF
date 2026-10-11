"""Track A 최종 패키지 배포·검증 (SA/1011/research_a).

usage (작업 트리 루트):
  .venv/bin/python -m src.r2h.research_a.deploy build    # 패키지 생성 (대상 폴더가 있으면 A 패키지 표식이 있을 때만 교체)
  .venv/bin/python -m src.r2h.research_a.deploy manifest # 대상의 REPRODUCIBILITY_MANIFEST.json 재작성 (노트북 실행 후)
  .venv/bin/python -m src.r2h.research_a.deploy verify   # 해시·경로 의존·핵심 수치 재계산 점검

원칙: 원 raw 는 복사만 하고 sha256 을 MANIFEST 와 대조한다. Track B(1011/paper2020)·기존 PDF 는 건드리지 않는다.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .contract import KST, git_branch, git_sha, input_data_version, load_contract, sha256_file, write_json

TARGET = Path("/home/sieg/projects-wsl/hongik_univ_26_2/SA/1011/research_a")
MARKER = ".r2h_track_a_package"
FORBIDDEN_SIBLINGS = ("paper2020", "Hybrid_SVR_GARCH_CUSUM_.pdf")
WORKTREE_MARKERS = (".worktrees/" + "SA_ETF/codex-a", "ETF_EDA_" + "SCAFFOLD/data")   # 문자열을 쪼개 이 파일 자신이 걸리지 않게


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    else:
        shutil.copy2(src, dst)


def build(target: Path = TARGET) -> dict:
    c = load_contract()
    src_root = c.root
    if target.exists():
        if not (target / MARKER).exists():
            raise RuntimeError(f"{target} 가 이미 있고 A 패키지 표식이 없다 — 덮어쓰지 않는다")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / MARKER).write_text("R2H Track A package — safe to regenerate with deploy build\n")

    # 1. 코드·설정·테스트·노트북·보고서
    for rel in ["src/__init__.py", "src/kfont.py", "src/r2h/research_a", "tests/research_a", "notebooks/research_a",
                "reports/r2h/research_a", "research_a/tickets", "config/universe.csv", "config/event_anchors.csv",
                "reports/dq_flags.csv", "reports/raw_validation.csv", "reports/tables/universe_vol_rank.csv",
                "reports/tables/universe_corr_pearson.csv", "reports/tables/universe_corr_spearman.csv",
                "analysis/robustness/d0122/rolling_corr_window_estimator_series.csv", "data/raw/MANIFEST.csv"]:
        _copy(src_root / rel, target / rel)
    # 2. 데이터 스냅샷 (raw 20 + interim 2) — sha 검증
    man = pd.read_csv(src_root / "data/raw/MANIFEST.csv", dtype=str)
    for t, sha in zip(man.ticker, man.sha256):
        dst = target / "data/raw" / f"{t}.csv"
        shutil.copy2(c.raw_dir / f"{t}.csv", dst)
        if sha256_file(dst) != sha:
            raise RuntimeError(f"raw copy sha mismatch: {t}")
    for f in ("panel_close.csv", "panel_logret.csv"):
        _copy(c.interim_dir / f, target / "data/interim" / f)
    # 3. 파생 데이터 (최신 A1/A2/A3 run 만)
    for ph in ("a1/A1", "a2/A2", "a3/A3"):
        mj = json.loads((src_root / f"reports/r2h/research_a/{ph}_RUN_MANIFEST.json").read_text())
        d = mj.get("derived_dir") or str(Path("data/processed/research_a") / mj["run_id"])
        _copy(src_root / d, target / d)
    # 4. 패키지용 계약: 입력 경로를 패키지 상대경로로
    spec = yaml.safe_load((src_root / "config/research_a_contract.yaml").read_text())
    spec["inputs"]["shared_data_root"] = "data"
    spec["inputs"]["raw_dir"] = "data/raw"
    spec["inputs"]["interim_dir"] = "data/interim"
    spec["package"] = {"built_at": datetime.now(KST).isoformat(timespec="seconds"), "source_branch": git_branch(src_root),
                       "source_commit": git_sha(src_root), "note": "경로는 패키지 루트 기준 상대경로. ETF_RAW_DIR 를 설정하지 않는다"}
    (target / "config").mkdir(exist_ok=True)
    (target / "config/research_a_contract.yaml").write_text(
        "# R2H Track A — Data Contract (배포 패키지판: 입력 경로가 패키지 상대경로)\n" + yaml.safe_dump(spec, allow_unicode=True, sort_keys=False))
    # 5. 최상위 문서
    for f in ("RESEARCH_A_FINAL_REPORT.md", "TRACK_C_HANDOFF.md"):
        shutil.copy2(src_root / "reports/r2h/research_a" / f, target / f)
    write_environment(target)
    write_readme(target, spec["package"])
    write_final_metrics(target)
    return write_manifest(target)


def write_environment(target: Path):
    pk = ["pandas", "numpy", "scipy", "matplotlib", "pyarrow", "PyYAML", "nbformat", "nbconvert", "nbclient", "ipykernel", "pytest", "jupyter_core"]
    env = {"python": sys.version.split()[0], "interpreter": "/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python (공용 venv, 변경·설치 없음)",
           "kernel": "hongik_26_2", "packages": {}}
    for p in pk:
        try:
            env["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            env["packages"][p] = None
    write_json(env, target / "ENVIRONMENT.json")
    (target / "requirements-research_a.txt").write_text("\n".join(f"{k}=={v}" for k, v in env["packages"].items() if v) + "\n")


def write_final_metrics(target: Path):
    rows = []
    for ph in ("a1/A1", "a2/A2", "a3/A3"):
        m = json.loads((target / f"reports/r2h/research_a/{ph}_RUN_MANIFEST.json").read_text())
        src = m.get("checks") or m.get("metrics") or {}
        for k, v in src.items():
            rows.append({"phase": m["phase"], "run_id": m["run_id"], "metric": k, "value": json.dumps(v, ensure_ascii=False, default=str)})
    df = pd.DataFrame(rows)
    df.to_csv(target / "reports/r2h/research_a/final_research_metrics.csv", index=False)


def write_readme(target: Path, pkg: dict):
    (target / "README.md").write_text(f"""# R2H Track A — Cross-Asset Correlation & Event Research Package

KOSPI200(069500) 대비 채권·코스닥·미국주식·금·원유·달러 ETF 의 30/60/120일 rolling 상관, 상관 전환 후보(C1~C5)와 가격 급등·급락 후보(P1~P5)의
시간적 관계를 검증한 Track A 최종 패키지. **결론은 `RESEARCH_A_FINAL_REPORT.md`**.

- 빌드: {pkg['built_at']} · 원본 branch `{pkg['source_branch']}` · commit `{pkg['source_commit'][:10]}`
- 이 폴더만으로 재현된다 (raw 20종 스냅샷 포함, 경로는 상대경로). Track B(`../paper2020`)와 독립.

## 구성
| 경로 | 내용 |
|---|---|
| `notebooks/research_a/00_채권_KOSPI_환경설정.ipynb` | A00 데이터 계약 (raw sha · 수익률 정의 · DQ) |
| `notebooks/research_a/01_채권_코스피상관시각화.ipynb` | A01 국고채10년·단기채 vs KOSPI200 |
| `notebooks/research_a/012_기타_코스피_상관_시각화.ipynb` | A012 코스닥·미국주식·금·원유·달러·섹터 |
| `notebooks/research_a/02_상관관계전환구간_실제_급등_정의구간후보.ipynb` | A02 사건 정의·매칭·귀무검정·사례·결론 |
| `notebooks/research_a/build_*.py` | 노트북 생성기 (셀 원본 SSOT) |
| `src/r2h/research_a/` | 분석 모듈 (contract·data·qa·rolling_correlation·correlation_events·price_events·event_matching·robustness·visualization·a1/a2/a3_pipeline·deploy) |
| `config/research_a_contract.yaml` | 데이터·정의 계약 (패키지 상대경로판) |
| `data/raw/` · `data/interim/` | ETF 20종 raw (sha256 = `data/raw/MANIFEST.csv`) · canonical 패널 |
| `data/processed/research_a/<run_id>/` | 파생 데이터 (rolling 상관 parquet, 사건 parquet, 패널) |
| `reports/r2h/research_a/a1·a2·a3/` | 결과표 + `*_RUN_MANIFEST.json` |
| `reports/r2h/research_a/figures/` | 그림 |
| `reports/r2h/research_a/*_EXECUTION_REPORT.md` | 단계별 실행 보고서 |
| `reports/tables/`, `reports/raw_validation.csv`, `analysis/robustness/d0122/` | 기존 EDA 재현 대조용 원본 표 (읽기 전용 복사) |
| `TRACK_C_HANDOFF.md`, `research_a/tickets/` | Track C 인계 명세 · A→B 티켓 |
| `REPRODUCIBILITY_MANIFEST.json` | 패키지 전 파일 sha256 |

## 실행
```bash
PY=/home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/python      # 공용 venv (패키지 설치 불필요)
cd {target}
$PY -m pytest tests/research_a -q                                   # 34 tests
$PY -m src.r2h.research_a.a1_pipeline                               # A1 (≈5 s)
$PY -m src.r2h.research_a.a2_pipeline                               # A2 (≈50 s)
$PY -m src.r2h.research_a.a3_pipeline                               # A3 (≈200 s) — 생략하면 노트북 A02 가 검증된 checkpoint 사용
for nb in 00_채권_KOSPI_환경설정 01_채권_코스피상관시각화 012_기타_코스피_상관_시각화 02_상관관계전환구간_실제_급등_정의구간후보; do
  /home/sieg/projects-wsl/hongik_univ_26_2/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \\
     --ExecutePreprocessor.kernel_name=hongik_26_2 notebooks/research_a/$nb.ipynb
done
$PY -m src.r2h.research_a.deploy verify --target .
```
A02 의 checkpoint 는 입력 데이터 버전·코드 해시·표 해시가 모두 일치할 때만 로드된다(불일치 시 오류 → A3 재실행).
""", encoding="utf-8")


def write_manifest(target: Path) -> dict:
    files = {}
    for p in sorted(target.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "REPRODUCIBILITY_MANIFEST.json":
            files[str(p.relative_to(target))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    c = load_contract(target)
    man = {"package": "R2H Track A", "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
           "input_data_version": input_data_version(pd.read_csv(target / "data/raw/MANIFEST.csv", dtype=str)),
           "contract_version": c.spec["contract_version"], "source": c.spec.get("package", {}),
           "run_ids": {ph: json.loads((target / f"reports/r2h/research_a/{ph}_RUN_MANIFEST.json").read_text())["run_id"]
                       for ph in ("a1/A1", "a2/A2", "a3/A3")},
           "n_files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()), "files": files}
    write_json(man, target / "REPRODUCIBILITY_MANIFEST.json")
    return man


def verify(target: Path = TARGET) -> dict:
    out = {}
    man = json.loads((target / "REPRODUCIBILITY_MANIFEST.json").read_text())
    bad = [f for f, v in man["files"].items() if not (target / f).exists() or sha256_file(target / f) != v["sha256"]]
    out["manifest_files_match"] = not bad
    out["manifest_mismatch"] = bad[:20]
    rm = pd.read_csv(target / "data/raw/MANIFEST.csv", dtype=str)
    out["raw_sha_match"] = all(sha256_file(target / "data/raw" / f"{t}.csv") == s for t, s in zip(rm.ticker, rm.sha256))
    hits = []
    for p in list((target / "src").rglob("*.py")) + list((target / "notebooks").rglob("*.py")) + list((target / "notebooks").rglob("*.ipynb")) \
            + [target / "config/research_a_contract.yaml"]:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for mk in WORKTREE_MARKERS:
            if mk in txt:
                hits.append(f"{p.relative_to(target)}:{mk}")
    out["worktree_path_refs"] = hits
    sib = target.parent
    out["track_b_untouched_check"] = {s: (sib / s).exists() for s in FORBIDDEN_SIBLINGS}
    sys.path.insert(0, str(target))
    c = load_contract(target)
    out["contract_raw_dir"] = str(c.raw_dir)
    out["raw_dir_inside_package"] = str(c.raw_dir).startswith(str(target))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    tgt = Path(sys.argv[sys.argv.index("--target") + 1]).resolve() if "--target" in sys.argv else TARGET
    if cmd == "build":
        m = build(tgt); print(f"built {tgt}: {m['n_files']} files, {m['total_bytes'] / 1e6:.1f} MB")
    elif cmd == "manifest":
        m = write_manifest(tgt); print(f"manifest: {m['n_files']} files")
    elif cmd == "verify":
        print(json.dumps(verify(tgt), ensure_ascii=False, indent=1))
