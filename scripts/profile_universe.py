"""ETF 별 profile json (notebook §12 출력) → reports/universe_profile.csv (ETF 당 1행, slot 순).

profile 이 없는 ETF 는 행을 남기되 profile_status=MISSING 으로 표시한다 (조용히 빠지지 않게).
usage: python scripts/profile_universe.py
"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "data" / "interim" / "profiles"


def main():
    u = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str, keep_default_na=False)
    rows = []
    for _, r in u.iterrows():
        p = PROFILE_DIR / f"{r['ticker']}.json"
        base = {"slot": int(r["slot"]), "ticker": r["ticker"], "name": r["name"]}
        if not p.exists():
            rows.append(base | {"profile_status": "MISSING"})
            continue
        prof = json.loads(p.read_text(encoding="utf-8"))
        flat = {k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in prof.items()
                if not isinstance(v, dict)}
        for k, v in prof.items():  # 중첩 dict (예: 기간별 통계) 는 key_subkey 로 펼친다
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if not isinstance(vv, (dict, list)):
                        flat[f"{k}__{kk}"] = vv
        rows.append(base | flat | {"profile_status": "OK"})
    df = pd.DataFrame(rows).sort_values("slot")
    out = ROOT / "reports" / "universe_profile.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} rows ({int((df['profile_status'] == 'OK').sum())} OK) × {df.shape[1]} cols → {out}")


if __name__ == "__main__":
    main()
