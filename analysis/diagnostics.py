from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def residual_diagnostics(residuals: pd.DataFrame) -> pd.DataFrame:
    """Compute residual moments and normality diagnostics by fitted model."""

    if residuals.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (portfolio, model_type, factor_model), group in residuals.groupby(
        ["portfolio", "model_type", "factor_model"]
    ):
        r = group["residual"].dropna().to_numpy(float)
        if len(r) < 8:
            jb_stat, jb_p = np.nan, np.nan
        else:
            jb = stats.jarque_bera(r)
            jb_stat, jb_p = float(jb.statistic), float(jb.pvalue)
        rows.append(
            {
                "portfolio": portfolio,
                "model_type": model_type,
                "factor_model": factor_model,
                "residual_mean": float(np.mean(r)) if len(r) else np.nan,
                "residual_std": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                "residual_skewness": float(stats.skew(r, bias=False))
                if len(r) > 2
                else np.nan,
                "residual_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False))
                if len(r) > 3
                else np.nan,
                "jarque_bera_stat": jb_stat,
                "jarque_bera_p_value": jb_p,
            }
        )
    return pd.DataFrame(rows)


def model_diagnostics(
    static_scores: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    runtime_seconds: float | None = None,
) -> pd.DataFrame:
    """Combine model fit metadata with residual diagnostics."""

    diag = static_scores.copy()
    if not residuals.empty:
        resid_diag = residual_diagnostics(residuals)
        diag = diag.merge(
            resid_diag,
            on=["portfolio", "model_type", "factor_model"],
            how="left",
            suffixes=("", "_diagnostic"),
        )

    diag["number_of_portfolios"] = int(static_scores["portfolio"].nunique())
    diag["number_of_observations"] = int(static_scores["n_obs"].sum())
    if runtime_seconds is not None:
        diag["pipeline_runtime_seconds"] = float(runtime_seconds)

    keep = [
        "portfolio",
        "model_type",
        "factor_model",
        "sample_start",
        "sample_end",
        "number_of_portfolios",
        "number_of_observations",
        "n_obs",
        "log_likelihood",
        "AIC",
        "BIC",
        "sigma_v",
        "sigma_u",
        "lambda",
        "converged",
        "runtime_seconds",
        "pipeline_runtime_seconds",
        "residual_mean",
        "residual_std",
        "residual_skewness",
        "residual_kurtosis",
        "jarque_bera_p_value",
    ]
    existing = [col for col in keep if col in diag.columns]
    return diag[existing].sort_values("portfolio").reset_index(drop=True)
