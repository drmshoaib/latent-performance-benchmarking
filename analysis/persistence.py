from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_persistence(
    df: pd.DataFrame,
    horizon: int = 1,
    date_col: str = "window_end",
    value_col: str = "AE_H1_mean",
) -> pd.DataFrame:
    """
    Compute Spearman rank persistence across rolling windows.

    Parameters
    ----------
    horizon : int
        Number of rolling steps ahead (1 = next window).
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

   # --- CRITICAL FIX ---
   # Enforce one observation per (date, portfolio)
    df = (
    df.groupby([date_col, "portfolio"], as_index=False)[value_col]
      .mean()
)

    df = df.sort_values([date_col, "portfolio"])


    dates = sorted(df[date_col].unique())
    results = []

    for i in range(len(dates) - horizon):
        t0 = dates[i]
        t1 = dates[i + horizon]

        d0 = df[df[date_col] == t0].set_index("portfolio")[value_col]
        d1 = df[df[date_col] == t1].set_index("portfolio")[value_col]

        common = d0.index.intersection(d1.index)
        if len(common) < 5:
            continue

        rho, _ = spearmanr(d0.loc[common], d1.loc[common])

        results.append(
            {
                "date_t": t0,
                "date_t_plus_h": t1,
                "horizon": horizon,
                "spearman_rho": rho,
            }
        )

    return pd.DataFrame(results)
