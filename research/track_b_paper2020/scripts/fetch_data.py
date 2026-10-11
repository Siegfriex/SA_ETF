"""Fetch the three §5 series. Investing.com (paper source) returned HTTP 403 on
2026-10-11, so Yahoo Finance's public chart endpoint is used (SOURCE_SUBSTITUTED).

Writes data/raw/<name>.csv (date, close) and data/raw/data_manifest.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SERIES = {
    # name: (yahoo symbol, paper start, paper end, meaning)
    "sp500": ("^GSPC", "2012-01-03", "2016-09-30", "S&P 500 index level"),
    "kospi": ("^KS11", "2012-10-02", "2019-06-28", "KOSPI composite index level"),
    "krw_usd": ("KRW=X", "2012-01-02", "2016-09-30", "KRW per 1 USD (USD/KRW quote)"),
}
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


def _epoch(day: str, plus_days: int = 0) -> int:
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp()) + plus_days * 86400


def fetch(name: str) -> dict:
    sym, start, end, meaning = SERIES[name]
    params = {"period1": _epoch(start, -3), "period2": _epoch(end, 3), "interval": "1d"}
    r = requests.get(URL.format(sym=sym), params=params, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz_offset = res["meta"].get("gmtoffset", 0)
    rows = {}
    for ts, close in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]):
        day = datetime.fromtimestamp(ts + tz_offset, tz=timezone.utc).strftime("%Y-%m-%d")
        if close is None or not (start <= day <= end):
            continue
        rows[day] = float(close)  # later duplicate on same local date overwrites
    path = RAW / f"{name}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "close"])
        for day in sorted(rows):
            w.writerow([day, repr(rows[day])])
    days = sorted(rows)
    return {
        "name": name, "meaning": meaning, "provider": "Yahoo Finance chart API v8",
        "symbol": sym, "url": r.url, "paper_source": "Investing.com",
        "label": "SOURCE_SUBSTITUTED", "price_field": "close (unadjusted index/FX close)",
        "paper_period": [start, end], "first_date": days[0], "last_date": days[-1],
        "n_levels": len(days), "n_null_dropped": sum(c is None for c in res["indicators"]["quote"][0]["close"]),
        "n_timestamps_raw": len(res["timestamp"]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"investing_com_probe": "HTTP 403 on https://www.investing.com/indices/us-spx-500-historical-data (2026-10-11); not retried",
                "series": {}}
    for name in SERIES:
        try:
            manifest["series"][name] = fetch(name)
        except Exception as exc:  # record and continue
            manifest["series"][name] = {"status": "BLOCKED", "error": repr(exc)}
        print(name, {k: manifest["series"][name].get(k) for k in ("first_date", "last_date", "n_levels", "status")})
    (RAW / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
