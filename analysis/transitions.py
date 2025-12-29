# analysis/transitions.py
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_quintile_transitions(
    df: pd.DataFrame,
    value_col: str = "AE_H1_mean",
    horizon: int = 1,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Compute cross-sectional quintile transition matrix.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: ['window_end', 'portfolio', value_col]
    value_col : str
        Latent performance metric (default: AE_H1_mean)
    horizon : int
        Forward horizon in windows
    n_quantiles : int
        Number of quantiles (default: 5)

    Returns
    -------
    DataFrame
        Transition probability matrix
    """

    df = df.copy()
    df = df.sort_values(["window_end", "portfolio"])

    # Assign quintiles cross-sectionally
    def assign_quantiles(x):
        return pd.qcut(
            x,
            q=n_quantiles,
            labels=False,
            duplicates="drop"
        ) + 1

    df["quintile"] = (
        df.groupby("window_end")[value_col]
        .transform(assign_quantiles)
    )

    # Shift forward by horizon
    df["quintile_fwd"] = (
        df.groupby("portfolio")["quintile"]
        .shift(-horizon)
    )

    # Drop incomplete transitions
    df = df.dropna(subset=["quintile", "quintile_fwd"])

    # Count transitions
    counts = (
        df.groupby(["quintile", "quintile_fwd"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert to probabilities (row-normalised)
    probs = counts.div(counts.sum(axis=1), axis=0)

    probs.index.name = "From Quintile"
    probs.columns.name = "To Quintile"

    return probs
