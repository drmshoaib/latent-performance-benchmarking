from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rolling_windows import rolling_sfa


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def rolling_window_sensitivity(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    factor_model: str,
    model_type: str = "half_normal",
    windows: tuple[int, ...] = (60, 120, 180),
    step: int = 12,
    min_obs: int | None = None,
    maxiter: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare rolling AE/rank stability across rolling-window lengths."""

    rolling_by_window: dict[int, pd.DataFrame] = {}
    first_end = max(windows)
    for window in windows:
        min_required = min_obs or window
        out = rolling_sfa(
            df,
            factor_cols,
            window=window,
            min_obs=min_required,
            step=step,
            model_type=model_type,
            factor_model=factor_model,
            maxiter=maxiter,
            first_end=first_end,
        )
        out["sensitivity_window"] = int(window)
        rolling_by_window[int(window)] = out

    rows: list[dict] = []
    for left_window, right_window in itertools.combinations(windows, 2):
        left = rolling_by_window[int(left_window)].dropna(subset=["AE", "rank"])
        right = rolling_by_window[int(right_window)].dropna(subset=["AE", "rank"])
        merged = left.merge(
            right,
            on=["portfolio", "window_end"],
            suffixes=(f"_{left_window}", f"_{right_window}"),
        )
        if merged.empty:
            rows.append(
                {
                    "window_left": left_window,
                    "window_right": right_window,
                    "n_common_observations": 0,
                    "rank_correlation": np.nan,
                    "AE_correlation": np.nan,
                    "top_quintile_jaccard": np.nan,
                    "bottom_quintile_jaccard": np.nan,
                }
            )
            continue

        rank_corr = spearmanr(
            merged[f"rank_{left_window}"], merged[f"rank_{right_window}"]
        ).statistic
        ae_corr = pearsonr(
            merged[f"AE_{left_window}"], merged[f"AE_{right_window}"]
        ).statistic

        top_scores = []
        bottom_scores = []
        for _, group in merged.groupby("window_end"):
            top_left = set(
                group.loc[
                    group[f"quintile_{left_window}"].astype(int) == 5, "portfolio"
                ]
            )
            top_right = set(
                group.loc[
                    group[f"quintile_{right_window}"].astype(int) == 5, "portfolio"
                ]
            )
            bottom_left = set(
                group.loc[
                    group[f"quintile_{left_window}"].astype(int) == 1, "portfolio"
                ]
            )
            bottom_right = set(
                group.loc[
                    group[f"quintile_{right_window}"].astype(int) == 1, "portfolio"
                ]
            )
            top_scores.append(_jaccard(top_left, top_right))
            bottom_scores.append(_jaccard(bottom_left, bottom_right))

        rows.append(
            {
                "window_left": int(left_window),
                "window_right": int(right_window),
                "n_common_observations": int(len(merged)),
                "rank_correlation": float(rank_corr),
                "AE_correlation": float(ae_corr),
                "top_quintile_jaccard": float(np.nanmean(top_scores)),
                "bottom_quintile_jaccard": float(np.nanmean(bottom_scores)),
            }
        )

    combined = pd.concat(rolling_by_window.values(), ignore_index=True)
    return pd.DataFrame(rows), combined


def model_comparison(
    half_normal_scores: pd.DataFrame,
    truncated_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compare static half-normal and truncated-normal SFA outputs."""

    left = half_normal_scores[
        ["portfolio", "AE", "AE_rank", "log_likelihood", "AIC", "BIC", "converged"]
    ].rename(
        columns={
            "AE": "AE_half_normal",
            "AE_rank": "AE_rank_half_normal",
            "log_likelihood": "log_likelihood_half_normal",
            "AIC": "AIC_half_normal",
            "BIC": "BIC_half_normal",
            "converged": "converged_half_normal",
        }
    )
    right = truncated_scores[
        ["portfolio", "AE", "AE_rank", "log_likelihood", "AIC", "BIC", "converged"]
    ].rename(
        columns={
            "AE": "AE_truncated_normal",
            "AE_rank": "AE_rank_truncated_normal",
            "log_likelihood": "log_likelihood_truncated_normal",
            "AIC": "AIC_truncated_normal",
            "BIC": "BIC_truncated_normal",
            "converged": "converged_truncated_normal",
        }
    )
    merged = left.merge(right, on="portfolio", how="inner")
    merged["AE_difference"] = merged["AE_half_normal"] - merged["AE_truncated_normal"]
    merged["rank_difference"] = (
        merged["AE_rank_half_normal"] - merged["AE_rank_truncated_normal"]
    )
    if len(merged) >= 3:
        merged["rank_spearman"] = spearmanr(
            merged["AE_rank_half_normal"], merged["AE_rank_truncated_normal"]
        ).statistic
    else:
        merged["rank_spearman"] = np.nan
    return merged
