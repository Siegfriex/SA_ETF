"""Figure 1 (all three panels): ACF/PACF up to lag 25 of the training log-returns
(first half of each series, paper §5) for S&P 500, KOSPI, KRW/USD (Yahoo substitute)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(3, 2, figsize=(10, 8))
for i, (name, title) in enumerate((("sp500", "(a) S&P 500"), ("kospi", "(b) KOSPI"), ("krw_usd", "(c) KRW/USD"))):
    lv = pd.read_csv(ROOT / "data" / "raw" / f"{name}.csv").close.to_numpy(float)
    r = 100 * np.diff(np.log(lv))
    tr = r[: r.size // 2]
    plot_acf(tr, lags=25, ax=ax[i, 0], zero=False, title=f"{title} ACF (n={tr.size})")
    plot_pacf(tr, lags=25, ax=ax[i, 1], zero=False, title=f"{title} PACF")
fig.suptitle("Figure 1 (replication, Yahoo substitute): training log-returns, first half")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig1_acf_pacf_all.png", dpi=120)
print("ok")
