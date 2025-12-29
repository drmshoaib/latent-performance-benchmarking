# analysis/mobility.py
from __future__ import annotations

import pandas as pd


def compute_mobility_metrics(
    transition_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute stay / improve / deteriorate probabilities
    from a quintile transition matrix.

    Rows must sum to 1.
    """

    # Ensure quintiles are treated as integers
    mat = transition_matrix.copy()
    mat.index = mat.index.astype(float).astype(int)
    mat.columns = mat.columns.astype(float).astype(int)

    results = []

    q_min = mat.index.min()
    q_max = mat.index.max()

    for q in mat.index:
        row = mat.loc[q]

        stay = row.loc[q]

        improve = row[row.index < q].sum() if q > q_min else 0.0
        deteriorate = row[row.index > q].sum() if q < q_max else 0.0

        results.append(
            {
                "Quintile": q,
                "Stay": float(stay),
                "Improve": float(improve),
                "Deteriorate": float(deteriorate),
            }
        )

    return pd.DataFrame(results)
