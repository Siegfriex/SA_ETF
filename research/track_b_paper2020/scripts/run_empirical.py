"""§5 empirical analysis: S&P 500, KOSPI, KRW/USD (Figures 1-4).

Writes results/empirical_summary.json, results/empirical_*.csv, figures/fig*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dataclasses  # noqa: E402
import paper2020.runner as runner  # noqa: E402
from paper2020.runner import empirical_analysis  # noqa: E402
from paper2020.svr_garch import TestVarianceInputPolicy  # noqa: E402

_base_make_spec = runner.make_spec

PAPER = {  # paper §5 targets (contract YAML)
    "sp500": {"date": "2015-08-18", "stat": 2.4751, "title": "S&P 500"},
    "kospi": {"date": "2018-01-22", "stat": 1.5081, "title": "KOSPI"},
    "krw_usd": {"date": "2014-09-26", "stat": 4.2754, "title": "KRW/USD"},
}


def main() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    summary = {}
    for k, (name, meta) in enumerate(PAPER.items(), start=2):
        df = pd.read_csv(ROOT / "data" / "raw" / f"{name}.csv", parse_dates=["date"])
        assert df.date.is_monotonic_increasing and not df.date.duplicated().any()
        levels = df.close.to_numpy(float)
        out = {}
        for lane, clip, policy in (("raise", False, "observed"), ("clip", True, "observed"),
                                   ("recursive_clip", True, "recursive")):
            if policy == "recursive":
                runner.make_spec = lambda kind="full", clip=False: dataclasses.replace(
                    _base_make_spec(kind, clip=clip),
                    test_variance_input_policy=TestVarianceInputPolicy.RECURSIVE_PREDICTION)
            else:
                runner.make_spec = _base_make_spec
            try:
                res = empirical_analysis(levels, clip=clip)
            except ValueError as exc:
                out[lane] = {"status": "failed", "error": str(exc)[:200]}
                continue
            cu = res["cusum"]
            dates = df.date.iloc[1:].reset_index(drop=True)
            cp_date = dates.iloc[res["change_return_index"]].strftime("%Y-%m-%d")
            out[lane] = {
                "status": "ok", "statistic": round(cu.statistic, 4), "reject_5pct": cu.reject,
                "change_date": cp_date, "n_returns": int(res["returns"].size), "n_test": int(res["residuals"].size),
                "fit_start": dates.iloc[0].strftime("%Y-%m-%d"),
                "fit_end": dates.iloc[res["half"] - 1].strftime("%Y-%m-%d"),
                "best_svr": vars(res["fit"].best_parameters),
                "n_nonpositive_variance": res["n_nonpositive_variance"],
                "max_abs_return": round(float(np.max(np.abs(res["returns"]))), 3),
                "date_max_abs_return": dates.iloc[int(np.argmax(np.abs(res["returns"])))].strftime("%Y-%m-%d"),
            }
            if lane == "raise" or "plot" not in out:
                out["plot"] = (res, dates, cp_date)
        plot = out.pop("plot", None)
        summary[name] = {"paper": meta, **out}
        print(name, json.dumps(out, default=str))
        if plot is None:
            continue
        res, dates, cp_date = plot
        r = res["returns"]
        if name == "sp500":  # Figure 1: ACF/PACF of training log-returns (lag 25)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
            plot_acf(r[: res["half"]], lags=25, ax=ax[0], zero=False)
            plot_pacf(r[: res["half"]], lags=25, ax=ax[1], zero=False)
            fig.suptitle("Figure 1 (replication): S&P 500 training log-returns")
            fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig1_sp500_acf_pacf.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
        ax[0].plot(df.date, df.close, lw=0.8)
        ax[0].set_ylabel("level")
        ax[1].plot(dates, r, lw=0.6)
        ax[1].set_ylabel("100·log-return")
        for a in ax:
            a.axvline(pd.Timestamp(cp_date), color="C3", ls="--", label=f"replication {cp_date}")
            a.axvline(pd.Timestamp(meta["date"]), color="C2", ls=":", label=f"paper {meta['date']}")
            a.axvline(dates.iloc[res["half"]], color="0.5", lw=0.6)
        ax[0].legend(loc="upper left", fontsize=8)
        ax[0].set_title(f"Figure {k} (replication, Yahoo substitute): {meta['title']}  "
                        f"T={res['cusum'].statistic:.4f} (paper {meta['stat']})")
        fig.tight_layout(); fig.savefig(ROOT / "figures" / f"fig{k}_{name}.png", dpi=130); plt.close(fig)
        pd.DataFrame({"date": dates.iloc[res["half"]:].to_numpy(), "return": r[res["half"]:],
                      "sigma2_hat": res["variance"], "residual": res["residuals"],
                      "centered_path": res["cusum"].centered_path}).to_csv(
            ROOT / "results" / f"empirical_{name}_test.csv", index=False)
    (ROOT / "results" / "empirical_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
