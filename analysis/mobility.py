from __future__ import annotations

import numpy as np
import pandas as pd

MOBILITY_COLUMNS = [
    "portfolio",
    "mean_rank",
    "median_rank",
    "rank_volatility",
    "mean_AE",
    "AE_volatility",
    "maximum_rank_improvement",
    "maximum_rank_deterioration",
    "same_quintile_probability",
    "move_up_probability",
    "move_down_probability",
    "time_in_quintile_1",
    "time_in_quintile_2",
    "time_in_quintile_3",
    "time_in_quintile_4",
    "time_in_quintile_5",
]


def portfolio_mobility_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    """Summarise portfolio-level rolling rank and quintile mobility."""

    required = {"portfolio", "window_end", "AE", "rank", "quintile"}
    missing = required - set(rolling.columns)
    if missing:
        raise ValueError(f"rolling data missing required columns: {sorted(missing)}")

    df = rolling.dropna(subset=["AE", "rank", "quintile"]).copy()
    if df.empty:
        return pd.DataFrame(columns=MOBILITY_COLUMNS)

    df["window_end"] = pd.to_datetime(df["window_end"])
    df = df.sort_values(["portfolio", "window_end"])
    df["rank"] = df["rank"].astype(float)
    df["quintile"] = df["quintile"].astype(int)
    df["next_quintile"] = df.groupby("portfolio")["quintile"].shift(-1)
    df["next_rank"] = df.groupby("portfolio")["rank"].shift(-1)
    df["rank_change"] = df["rank"] - df["next_rank"]

    rows: list[dict] = []
    for portfolio, group in df.groupby("portfolio"):
        transitions = group.dropna(subset=["next_quintile"])
        row = {
            "portfolio": portfolio,
            "mean_rank": float(group["rank"].mean()),
            "median_rank": float(group["rank"].median()),
            "rank_volatility": float(group["rank"].std(ddof=1)),
            "mean_AE": float(group["AE"].mean()),
            "AE_volatility": float(group["AE"].std(ddof=1)),
            "maximum_rank_improvement": float(np.nanmax(group["rank_change"]))
            if group["rank_change"].notna().any()
            else np.nan,
            "maximum_rank_deterioration": float(-np.nanmin(group["rank_change"]))
            if group["rank_change"].notna().any()
            else np.nan,
        }

        if transitions.empty:
            row.update(
                {
                    "same_quintile_probability": np.nan,
                    "move_up_probability": np.nan,
                    "move_down_probability": np.nan,
                }
            )
        else:
            row.update(
                {
                    "same_quintile_probability": float(
                        np.mean(transitions["next_quintile"] == transitions["quintile"])
                    ),
                    "move_up_probability": float(
                        np.mean(transitions["next_quintile"] > transitions["quintile"])
                    ),
                    "move_down_probability": float(
                        np.mean(transitions["next_quintile"] < transitions["quintile"])
                    ),
                }
            )

        shares = group["quintile"].value_counts(normalize=True)
        for q in range(1, 6):
            row[f"time_in_quintile_{q}"] = float(shares.get(q, 0.0))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=MOBILITY_COLUMNS)
    return (
        pd.DataFrame(rows)[MOBILITY_COLUMNS]
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def compute_mobility_metrics(transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible transition-matrix mobility summary."""

    mat = transition_matrix.copy()
    mat.index = mat.index.astype(float).astype(int)
    mat.columns = mat.columns.astype(float).astype(int)

    rows = []
    for q in mat.index:
        row = mat.loc[q]
        rows.append(
            {
                "Quintile": q,
                "Stay": float(row.loc[q]),
                "Improve": float(row[row.index > q].sum()),
                "Deteriorate": float(row[row.index < q].sum()),
            }
        )
    return pd.DataFrame(rows)
