from __future__ import annotations

import numpy as np
import pandas as pd

from sfa.sfa_halfnormal import HalfNormalSFA
from sfa.sfa_truncated import TruncatedNormalSFA


def rolling_sfa(
    df: pd.DataFrame,
    window: int = 120,
    min_obs: int = 120,
    step: int = 12,   # NEW: yearly stride
) -> pd.DataFrame:

    """
    Rolling-window SFA efficiency estimation.

    Parameters
    ----------
    df : DataFrame
        Columns:
        date, portfolio, excess_return, mkt_rf, smb, hml
    window : int
        Window length in months (default = 120)
    min_obs : int
        Minimum observations required per window

    Returns
    -------
    DataFrame
        window_end, portfolio, AE_H1_mean
    """

    results = []

    for portfolio, g in df.groupby("portfolio"):
        g = g.sort_values("date").reset_index(drop=True)

        if len(g) < min_obs:
            continue

        for end in range(window, len(g) + 1, step):
            w = g.iloc[end - window : end]

            if len(w) < min_obs:
                continue

            y = w["excess_return"].to_numpy(float)
            X = np.column_stack(
                [
                    np.ones(len(w)),
                    w["mkt_rf"].to_numpy(float),
                    w["smb"].to_numpy(float),
                    w["hml"].to_numpy(float),
                ]
            )

            # ---- Half-normal ----
            try:
                hn = HalfNormalSFA(y, X).fit()
                ae_hn = hn.TE
            except Exception:
                ae_hn = np.full(len(w), np.nan)

            # ---- Truncated-normal ----
            try:
                tn = TruncatedNormalSFA(y, X).fit()
                ae_tn = tn.TE
            except Exception:
                ae_tn = np.full(len(w), np.nan)

            ae_h1 = np.nanmean(np.column_stack([ae_hn, ae_tn]), axis=1)

            results.append(
                {
                    "window_end": w["date"].iloc[-1],
                    "portfolio": portfolio,
                    "AE_H1_mean": float(np.nanmean(ae_h1)),
                }
            )

    return pd.DataFrame(results)
