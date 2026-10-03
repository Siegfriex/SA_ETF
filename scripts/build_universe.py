"""PROVISIONAL ETF universe 생성 (C-0001 slot plan 을 FDR listing 으로 검증).

- 모집단: fdr.StockListing("ETF/KR")  (실제 columns: Symbol, Category, Name, Price, RiseFall, Change,
  ChangeRate, NAV, EarningRate, Volume, Amount, MarCap — listing_date 없음)
- listing_date: listing 에 없으므로 fdr.DataReader(t, "2019-01-01") 첫 행 날짜를 data_start 로 기록.
  data_start == 2019-01-02 이면 "상장일 ≤ 2019-01-02" 만 알 수 있다 (listing_date_source 에 명시).
- taxonomy(asset_scope/category/benchmark/peer_group/...) 는 B 1차값 — A 가 FACT_CORRECTION 가능.
- 공식 수업 ticker list 가 오면 SLOT_PLAN 을 교체한다.

usage: python scripts/build_universe.py   → config/universe.csv, data/raw/listing_snapshot_*.csv(gitignored)
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys

import pandas as pd
import FinanceDataReader as fdr

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path("/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD/data/raw")
KST = timezone(timedelta(hours=9))

# FDR ETF/KR Category 코드 — 종목명으로 검증한 의미 (B 0004 bus 기록)
FDR_CATEGORY = {1: "국내시장지수", 2: "국내업종/테마", 3: "국내파생(레버리지/인버스)",
                4: "해외주식", 5: "원자재", 6: "채권", 7: "기타"}

# slot, ticker, asset_scope, category, benchmark, peer_group, active_passive, leverage_inverse, exposure
SLOT_PLAN = [
    (1, "069500", "domestic_equity", "broad_market", "", "KR_large_cap", "passive", "1x", "KOSPI200"),
    (2, "102110", "domestic_equity", "broad_market", "069500", "KR_large_cap", "passive", "1x", "KOSPI200 (브랜드 대조)"),
    (3, "229200", "domestic_equity", "broad_market", "069500", "KR_small_mid_cap", "passive", "1x", "KOSDAQ150"),
    (4, "122630", "domestic_equity", "leveraged", "069500", "KR_large_cap_geared", "passive", "2x", "KOSPI200 2x"),
    (5, "114800", "domestic_equity", "inverse", "069500", "KR_large_cap_geared", "passive", "-1x", "KOSPI200 -1x"),
    (6, "091230", "domestic_equity", "sector", "069500", "KR_sector", "passive", "1x", "반도체"),
    (7, "091170", "domestic_equity", "sector", "069500", "KR_sector", "passive", "1x", "은행"),
    (8, "091180", "domestic_equity", "sector", "069500", "KR_sector", "passive", "1x", "자동차"),
    (9, "305540", "domestic_equity", "theme", "069500", "KR_theme", "passive", "1x", "2차전지 테마"),
    (10, "143860", "domestic_equity", "sector", "069500", "KR_sector", "passive", "1x", "헬스케어/바이오"),
    (11, "117680", "domestic_equity", "sector", "069500", "KR_sector", "passive", "1x", "철강/소재"),
    (12, "161510", "domestic_equity", "strategy", "069500", "KR_strategy", "passive", "1x", "고배당 전략"),
    (13, "219480", "foreign_equity", "broad_market", "069500", "US_equity", "passive", "1x", "미국 S&P500 선물(환헤지)"),
    (14, "133690", "foreign_equity", "broad_market", "069500", "US_equity", "passive", "1x", "미국 나스닥100"),
    (15, "192090", "foreign_equity", "broad_market", "069500", "CN_equity", "passive", "1x", "중국 CSI300"),
    (16, "148070", "bond", "government_bond", "", "KR_bond", "passive", "1x", "국고채10년"),
    (17, "153130", "bond", "short_term_bond", "", "KR_bond", "passive", "1x", "단기채/MMF성"),
    (18, "132030", "commodity", "precious_metal", "", "commodity", "passive", "1x", "금 선물(환헤지)"),
    (19, "261220", "commodity", "energy", "", "commodity", "passive", "1x", "WTI 원유 선물(환헤지)"),
    (20, "261240", "currency", "fx", "", "fx", "passive", "1x", "미국달러 선물"),
]
# slot 12 대안: 279530 KODEX 고배당주 (MarCap 3,462억 < 161510 23,643억 → 161510 채택)

MIN_ROWS = 1500                 # 2019-01-02 ~ 현재 ≈ 1,900 거래일
DATA_START_MAX = "2019-01-31"   # 공통 시간축 확보: 2019-01 안에 시작해야 함
LOW_AMOUNT_MILLION = 1000       # Amount(백만원, 스냅샷) < 10억 → 저유동성 flag (제외 아님)


def main():
    now = datetime.now(KST)
    listing = fdr.StockListing("ETF/KR")
    expected = {"Symbol", "Category", "Name", "MarCap", "Amount"}
    missing = expected - set(listing.columns)
    if missing:
        sys.exit(f"[STOP] listing columns 변경: {missing} 없음. 실제={list(listing.columns)}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    snap = RAW_DIR / f"listing_snapshot_{now:%Y%m%dT%H%M}.csv"
    listing.to_csv(snap, index=False)
    L = listing.set_index("Symbol")

    rows = []
    for slot, t, scope, cat, bench, peer, ap, lev, expo in SLOT_PLAN:
        if t not in L.index:
            sys.exit(f"[STOP] {t} listing 에 없음 — 같은 exposure 대체 필요")
        r = L.loc[t]
        d = fdr.DataReader(t, "2019-01-01")
        first = d.index.min().date().isoformat()
        flags = []
        if len(d) < MIN_ROWS:
            flags.append(f"short_history(rows={len(d)})")
        if first > DATA_START_MAX:
            flags.append(f"late_start({first})")
        if int(r["Amount"]) < LOW_AMOUNT_MILLION:
            flags.append(f"low_liquidity(amount_snapshot={int(r['Amount'])}M KRW)")
        if flags and any(f.startswith(("short", "late")) for f in flags):
            sys.exit(f"[STOP] {t} 기준 미달 {flags} — 같은 exposure 대체 필요")
        name = str(r["Name"])
        rows.append({
            "slot": slot, "ticker": t, "name": name,
            "provider_brand": name.split()[0],
            "listing_date": f"<= {first}" if first <= "2019-01-02" else first,
            "listing_date_source": "fdr.DataReader(start=2019-01-01) first row (상장일 아님, 하한 proxy)",
            "data_start": first,
            "asset_scope": scope, "category": cat, "benchmark": bench, "peer_group": peer,
            "active_passive": ap, "leverage_inverse": lev, "exposure": expo,
            "raw_file": f"data/raw/{t}.csv",
            "universe_status": "PROVISIONAL",
            "selection_reason": f"C-0001 slot plan · exposure={expo} · FDR listing 검증 통과"
                                + (f" · flags={';'.join(flags)}" if flags else ""),
            "taxonomy_source": "B 1차값(C plan) · A FACT_CORRECTION 가능",
            "fdr_category": f"{int(r['Category'])}:{FDR_CATEGORY.get(int(r['Category']), '?')}",
            "marcap_snapshot_100m_krw": int(r["MarCap"]),
            "amount_snapshot_million_krw": int(r["Amount"]),
            "snapshot_date": now.date().isoformat(),
            "selection_bias_note": "현재 상장·현재 규모 기준 선정(survivorship, D-005)",
        })
    u = pd.DataFrame(rows)
    out = ROOT / "config" / "universe.csv"
    u.to_csv(out, index=False)
    print(f"universe {len(u)} rows → {out}\nlisting snapshot → {snap} ({len(listing)} ETF)")
    print(u[["slot", "ticker", "name", "asset_scope", "category", "data_start", "selection_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
