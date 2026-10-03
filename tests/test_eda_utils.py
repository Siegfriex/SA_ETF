"""Plain-assert tests: python tests/test_eda_utils.py (pytest 도 가능)."""
import sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda_utils import load_raw, engineer_basic  # noqa: E402


def _frame(n=301, seed=0, volume=True):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.005, n)
    r[-1] = np.log(1.30)  # +30% jump on last day
    close = 100 * np.exp(np.cumsum(r))
    d = pd.bdate_range("2019-01-02", periods=n)
    df = pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Open": close, "High": close * 1.01,
                       "Low": close * 0.99, "Close": close, "Change": 0.0})
    if volume:
        df["Volume"] = (1000 + rng.integers(0, 100, n)).astype(float)
    return df


def _load(df):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x.csv"; df.to_csv(p, index=False)
        return load_raw(p)


def test_jump_oracle():
    df, _ = _load(_frame())
    x, _ = engineer_basic(df)
    z_shift = x["ret_z20"].iloc[-1]
    self_incl = x["log_ret"].iloc[-1] / x["log_ret"].rolling(20).std().iloc[-1]
    assert z_shift > 50, z_shift
    assert self_incl <= 4.5, self_incl  # bound ~ (n-1)/sqrt(n) = 4.25 + mean effect
    assert bool(x["pos_z3"].iloc[-1]) and not bool(x["neg_z3"].iloc[-1]) and bool(x["abs_z3"].iloc[-1])


def test_volume_shift():
    f = _frame(); f.loc[f.index[-1], "Volume"] = 1e6
    df, _ = _load(f)
    x, flags = engineer_basic(df)
    assert flags["has_volume"]
    assert x["volume_ratio_20"].iloc[-1] > 500  # baseline excludes today
    med_prev = df["volume"].iloc[-21:-1].median()
    assert abs(x["volume_ratio_20"].iloc[-1] - 1e6 / med_prev) < 1e-6


def test_provenance_dup_and_bad_date():
    f = _frame(50)
    f = pd.concat([f, f.iloc[[10]]], ignore_index=True)
    f.loc[len(f)] = f.iloc[5]; f.loc[len(f) - 1, "Date"] = "not-a-date"
    df, prov = _load(f)
    assert prov["rows_raw"] == 52, prov
    assert prov["n_duplicate_dates"] == 1, prov
    assert prov["n_unparseable_dates"] == 1, prov
    assert prov["was_sorted"] is False
    assert prov["rows_after"] == 50 and len(df) == 50
    assert "조정가격 수익률(추정)" in prov["price_policy"]


def test_no_volume():
    df, prov = _load(_frame(volume=False))
    x, flags = engineer_basic(df)
    assert prov["has_volume"] is False and flags["has_volume"] is False
    assert "volume_robust_z_20" not in x.columns
    f = _frame(); f["Volume"] = 0.0
    df, prov = _load(f)
    x, flags = engineer_basic(df)
    assert flags["has_volume"] is False and flags["n_zero_volume_days"] == len(df)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("ALL PASS")
