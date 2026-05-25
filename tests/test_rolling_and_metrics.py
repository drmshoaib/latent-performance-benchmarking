from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.alpha_baseline import compare_alpha_to_ae, estimate_alpha_baseline
from analysis.mobility import portfolio_mobility_summary
from analysis.rolling_windows import _assign_window_ranks, rolling_sfa
from analysis.transitions import compute_transition_matrix


def test_rolling_windows_have_expected_metadata(synthetic_dataset):
    rolling = rolling_sfa(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        window=12,
        min_obs=12,
        step=12,
        factor_model="ff3",
        maxiter=40,
    )

    assert len(rolling) == 5 * 4
    assert {
        "window_start",
        "window_end",
        "window_length",
        "AE",
        "rank",
        "quintile",
    } <= set(rolling.columns)
    assert rolling["window_length"].eq(12).all()
    assert (rolling["window_end"] > rolling["window_start"]).all()
    assert rolling["AE"].between(0, 1, inclusive="right").all()


def test_insufficient_rolling_observations_return_empty(synthetic_dataset):
    rolling = rolling_sfa(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        window=120,
        min_obs=120,
        step=12,
    )

    assert rolling.empty


def test_rank_and_quintile_assignment_is_deterministic():
    df = pd.DataFrame(
        {
            "portfolio": [f"P{i}" for i in range(1, 6)],
            "window_end": pd.Timestamp("2020-12-31"),
            "AE": [0.9, 0.7, 0.8, 0.6, 0.95],
            "convergence_status": True,
        }
    )

    ranked = _assign_window_ranks(df)
    ordered = ranked.sort_values("rank")["portfolio"].tolist()

    assert ordered == ["P5", "P1", "P3", "P2", "P4"]
    assert set(ranked["quintile"].dropna().astype(int)) == {1, 2, 3, 4, 5}


def test_transition_matrix_shape_bounds_and_row_sums():
    rows = []
    for date in [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]:
        for q in range(1, 6):
            rows.append(
                {
                    "portfolio": f"P{q}",
                    "window_end": date,
                    "AE": q / 10,
                    "rank": 6 - q,
                    "quintile": q,
                }
            )
    rolling = pd.DataFrame(rows)

    matrix, summary = compute_transition_matrix(rolling, horizon_months=12)

    assert matrix.shape == (5, 5)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert ((matrix >= 0) & (matrix <= 1)).all().all()
    assert {"stay_probability", "upgrade_probability", "downgrade_probability"} <= set(
        summary.columns
    )


def test_mobility_summary_probabilities_and_rank_volatility():
    rows = []
    for idx, date in enumerate(pd.date_range("2020-12-31", periods=3, freq="YE")):
        for q in range(1, 6):
            rows.append(
                {
                    "portfolio": f"P{q}",
                    "window_end": date,
                    "AE": q / 10 + idx * 0.001,
                    "rank": 6 - q,
                    "quintile": min(5, q + (idx == 2 and q < 5)),
                }
            )
    mobility = portfolio_mobility_summary(pd.DataFrame(rows))

    expected = {
        "mean_rank",
        "rank_volatility",
        "same_quintile_probability",
        "move_up_probability",
        "move_down_probability",
    }
    assert expected <= set(mobility.columns)
    assert mobility["rank_volatility"].ge(0).all()
    for col in [
        "same_quintile_probability",
        "move_up_probability",
        "move_down_probability",
    ]:
        assert mobility[col].between(0, 1).all()


def test_alpha_baseline_and_alpha_ae_comparison(synthetic_dataset):
    alpha = estimate_alpha_baseline(
        synthetic_dataset.data,
        synthetic_dataset.factor_cols,
        factor_model="ff3",
        min_obs=12,
    )
    ae = pd.DataFrame(
        {
            "portfolio": alpha["portfolio"],
            "AE": np.linspace(0.99, 0.95, len(alpha)),
            "AE_rank": range(1, len(alpha) + 1),
            "u_hat": np.linspace(0.001, 0.005, len(alpha)),
            "converged": True,
        }
    )

    comparison = compare_alpha_to_ae(alpha, ae)

    assert {"alpha_rank", "AE_rank", "alpha_ae_spearman", "rank_difference"} <= set(
        comparison.columns
    )
    assert comparison["alpha_rank"].notna().all()
