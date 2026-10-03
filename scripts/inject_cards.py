"""실행된 notebook 에 ETF 고유 해석 Markdown 을 주입한다 (C-0018). 재실행 없음 — nbformat 으로 Markdown 만 추가.

주입 내용 (섹션별, 캡션 셀 바로 뒤):
1. "WHAT WE SEE — 이 ETF (universe 비교)": universe_profile.csv 수치를 문장으로 + 20종 내 순위·백분위·069500 대비 배수 (B, 결정적 계산)
2. "ETF-specific reading (A card @sha)": A 의 reports/ETF_CARDS/{slot}_{ticker}.md 해당 섹션 원문. 카드 없으면 "카드 미도착" 명시
3. DQ 각주: reports/dq_flags.csv 의 해당 ETF 행
그리고 기존 캡션 출력의 generic 문단(WHY / NOT KNOW / NEXT) heading 을 "General note" 로 낮춘다.

멱등: 주입 셀은 metadata.b_inject=True 로 표시, 재실행 시 지우고 다시 넣는다. 출력 heading 치환도 멱등.
usage: python scripts/inject_cards.py [--card-ref origin/claude-a/eda-analysis-v1]
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import re
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks" / "00_raw_eda"
KST = timezone(timedelta(hours=9))
REF = "069500"
CARD_REF = sys.argv[sys.argv.index("--card-ref") + 1] if "--card-ref" in sys.argv else "origin/claude-a/eda-analysis-v1"

# card section heading → notebook section(s)
CARD_MAP = {"00": ["정체성"], "03": ["평소 움직임"], "04": ["변동성 구조"], "05": ["거래활동"],
            "06": ["언제 이상해지는가"], "07": ["언제 이상해지는가"], "08": ["peer 와 다른 점"],
            "11": ["WHAT WE SEE / WHY IT MATTERS / WHAT WE DO NOT KNOW / NEXT QUESTION", "Hypothesis candidates"]}
DD_WORDS = ("max_dd", "drawdown", "회복", "낙폭", "dd ")

# section → [(profile column, 한국어 이름, 형식, 높은 순 순위면 True)]
METRICS = {
    "01": [("n_obs", "관측일수", "{:,.0f}", True)],
    "03": [("ann_vol", "연환산 변동성", "{:.1%}", True), ("ex_kurt", "초과첨도(full)", "{:.1f}", True),
           ("ex_kurt_trim3", "초과첨도(상위 |r| 3일 제외)", "{:.1f}", True), ("skew", "왜도", "{:+.2f}", True),
           ("p01", "일간 로그수익률 1% 분위", "{:+.2%}", False), ("p99", "일간 로그수익률 99% 분위", "{:+.2%}", True)],
    "04": [("acf_absret_lag1", "|r| lag1 자기상관(변동성 군집)", "{:.3f}", True),
           ("vol_of_vol", "vol-of-vol", "{:.3f}", True)],
    "05": [("zero_vol_ratio", "거래량 0 일 비율", "{:.2%}", True)],
    "06": [("max_dd", "최대 낙폭", "{:.1%}", False), ("recovery_days", "최대 낙폭 회복일수", "{:,.0f}", True)],
    "07": [("n_abs_z3", "|z|≥3 후보일(shift(1) 20D vol 기준)", "{:,.0f}", True),
           ("n_pos_z3", "z≥+3 후보일", "{:,.0f}", True), ("n_neg_z3", "z≤−3 후보일", "{:,.0f}", True)],
    "08": [("corr_vs_benchmark", "benchmark proxy 와 상관", "{:.3f}", True),
           ("beta_vs_benchmark", "benchmark proxy 대비 beta", "{:.2f}", True)],
    "09": [("pca_pc1_ratio", "PCA PC1 설명비율", "{:.1%}", True)],
}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def load_cards():
    sha = git("rev-parse", "--short", CARD_REF).stdout.strip()
    names = git("ls-tree", "--name-only", CARD_REF, "reports/ETF_CARDS/").stdout.split()
    cards = {}
    for n in names:
        m = re.search(r"(\d{2})_(\w{6})\.md$", n)
        if m:
            cards[m.group(2)] = git("show", f"{CARD_REF}:{n}").stdout
    return sha, cards


def sections(md):
    out, cur = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def rank_sentence(prof, t, col, label, fmt, high_first):
    s = pd.to_numeric(prof[col], errors="coerce")
    v = s.get(t, np.nan)
    if pd.isna(v):
        return f"- {label}: 값 없음 (해당 없음 또는 계산 불가)."
    n = int(s.notna().sum())
    rk = int(s.rank(ascending=not high_first, method="min")[t])
    pct = (s <= v).mean() * 100 if high_first else (s >= v).mean() * 100
    order = "높은 순" if high_first else "낮은(더 극단) 순"
    ref = s.get(REF, np.nan)
    rel = ""
    if t != REF and pd.notna(ref) and ref not in (0,) and np.sign(ref) == np.sign(v):
        rel = f", {REF} 의 {v / ref:.2f}배"
    med = s.median()
    return (f"- **{label} {fmt.format(v)}** — {n}종 중 {rk}위({order}), 백분위 {pct:.0f}%{rel} "
            f"(universe 중앙값 {fmt.format(med)}).")


def dq_note(dq, t):
    d = dq[dq["ticker"] == t]
    if d.empty:
        return "- DQ 각주: reports/dq_flags.csv 에 이 ETF 의 flag 없음."
    parts = []
    for cls, g in d.groupby("dq_class"):
        dates = sorted(g["date"])
        shown = ", ".join(dates[:6]) + (f" 외 {len(dates) - 6}일" if len(dates) > 6 else "")
        policy = {"price_anomaly_unverified": "통계는 해당일 포함/제외 병기(§03b dq_excluded)",
                  "ohlc_close_gt_high": "원인 UNRESOLVED, clip 없음 — range_pct 해석 주의",
                  "spike_reversal_candidate": "고립 급등-반전 후보, DQ 미확정 — 실제 사건일 수 있음"}.get(cls, "")
        parts.append(f"- **DQ 각주 · {cls}** ({len(dates)}일): {shown}. {policy}")
    return "\n".join(parts)


def md_cell(text, sec):
    c = nbformat.v4.new_markdown_cell(text)
    c.metadata["b_inject"] = True
    c.metadata["b_inject_section"] = sec
    return c


def demote_outputs(cell):
    for o in cell.get("outputs", []):
        md = o.get("data", {}).get("text/markdown")
        if md is None:
            continue
        s = "".join(md) if isinstance(md, list) else md
        s = re.sub(r"^#### (\d\d) · WHAT WE SEE$", r"#### \1 · WHAT WE SEE (raw values)", s, flags=re.M)
        for h in ("WHY IT MATTERS", "WHAT WE DO NOT KNOW", "NEXT QUESTION"):
            s = re.sub(rf"^#### {h}$", f"#### General note · {h}", s, flags=re.M)
        s = s.replace("> A/C 해석 통합 대기 (ETF_CARDS)", "> ETF 고유 해석은 바로 아래 'ETF-specific reading (A card)' 블록 참조")
        o["data"]["text/markdown"] = s


def main():
    prof = pd.read_csv(ROOT / "reports" / "universe_profile.csv", dtype={"ticker": str}).set_index("ticker")
    u = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False).set_index("ticker")
    dqp = ROOT / "reports" / "dq_flags.csv"
    dq = pd.read_csv(dqp, dtype=str, keep_default_na=False) if dqp.exists() else pd.DataFrame(columns=["ticker", "date", "dq_class"])
    card_sha, cards = load_cards()
    log = []
    for t, row in u.iterrows():
        slot = int(row["slot"])
        path = NB_DIR / f"00_{slot:02d}_{t}_raw_eda.ipynb"
        nb = nbformat.read(path, as_version=4)
        nb.cells = [c for c in nb.cells if not c.metadata.get("b_inject")]
        card = sections(cards[t]) if t in cards else None
        # insertion anchors: caption cell index per section; 00 = identity code cell (first code after "## 00")
        anchors = {}
        for i, c in enumerate(nb.cells):
            src = c.source if isinstance(c.source, str) else "".join(c.source)
            m = re.match(r'cap\("(\d\d)"', src.strip())
            if c.cell_type == "code" and m:
                if m.group(1) == "03" and "03" in anchors:  # 03b 가 두 번째 cap("03") → 뒤쪽(변형표) 뒤에 둔다
                    anchors["03"] = i
                else:
                    anchors.setdefault(m.group(1), i)
                demote_outputs(c)
            if c.cell_type == "markdown" and src.startswith("## 00 "):
                anchors["00"] = i + 1
            if c.cell_type == "markdown" and src.startswith("## 11 "):
                anchors["11"] = i + 1
            if c.cell_type == "code" and "A/C 해석 통합 대기" in json.dumps(c.get("outputs", []), ensure_ascii=False):
                demote_outputs(c)
        inserts = {}
        for sec in sorted(set(anchors) | set(CARD_MAP)):
            if sec not in anchors:
                continue
            parts = []
            if sec == "00":
                parts.append(f"### 00 · 이 ETF 는 무엇인가 (ETF-specific)\n"
                             f"- **{row['name']} ({t})** · {row['asset_scope']} / {row['category']} · 추종 {row['benchmark']} · "
                             f"레버리지 {row['leverage_inverse']} · 환헤지 {row['currency_hedge']} · 복제 {row['replication']}.\n"
                             f"- 거래시간: {row['trading_hours_note']}.\n"
                             f"- 비교 기준 ETF(benchmark proxy): {row['benchmark_proxy_ticker'] or '없음 — ' + row.get('benchmark_proxy_reason', '')}.\n"
                             f"- UNIVERSE_STATUS=PROVISIONAL · 수익률은 조정가격 수익률(추정, DEC-3).")
            if sec in METRICS:
                lines = [rank_sentence(prof, t, *m) for m in METRICS[sec] if m[0] in prof.columns]
                parts.append(f"### {sec} · WHAT WE SEE — 이 ETF (universe 20종 비교, B 계산)\n" + "\n".join(lines))
            if sec in ("01", "03", "07"):
                parts.append(dq_note(dq, t))
            if sec in CARD_MAP:
                if card is None:
                    parts.append(f"### ETF-specific reading (A card)\n> **카드 미도착** — reports/ETF_CARDS/{slot:02d}_{t}.md 가 "
                                 f"{CARD_REF}@{card_sha} 에 없음. 다음 inject 실행 시 채워진다.")
                else:
                    body = []
                    for h in CARD_MAP[sec]:
                        txt = card.get(h, "")
                        if sec == "06" and txt:
                            txt = "\n".join(l for l in txt.splitlines() if any(w in l for w in DD_WORDS)) or txt
                        if sec == "07" and txt:
                            txt = "\n".join(l for l in txt.splitlines() if not any(w in l for w in DD_WORDS)) or txt
                        body.append(f"**{h}**\n\n{txt}" if txt else f"**{h}**\n\n> 카드에 이 섹션 없음")
                    parts.append(f"### ETF-specific reading (A card @{card_sha})\n" + "\n\n".join(body))
            if parts:  # 주입할 내용이 없는 섹션(02 등)은 빈 셀을 만들지 않는다
                inserts[anchors[sec]] = md_cell("\n\n".join(parts), sec)
        for idx in sorted(inserts, reverse=True):  # anchor 셀(캡션/정체성/요약 코드) 바로 뒤
            nb.cells.insert(idx + 1, inserts[idx])
        nbformat.validate(nb)
        nbformat.write(nb, path)
        log.append({"ticker": t, "card": "present" if card else "missing", "n_injected": len(inserts)})
        print(t, "card" if card else "NO CARD", len(inserts), "cells")
    mp = ROOT / "reports" / "RUN_MANIFEST.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["inject"] = {"at": datetime.now(KST).isoformat(timespec="seconds"), "script": "scripts/inject_cards.py",
                   "card_ref": CARD_REF, "card_sha": card_sha, "rerun": False,
                   "cards_present": sum(l["card"] == "present" for l in log), "per_etf": log}
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print("card_sha", card_sha, "· cards", m["inject"]["cards_present"], "/", len(log))


if __name__ == "__main__":
    main()
