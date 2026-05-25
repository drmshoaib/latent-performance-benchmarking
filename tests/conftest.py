from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfa.loaders import build_factor_dataset

PORTFOLIOS = [
    "SMALL LoBM",
    "ME1 BM2",
    "ME1 BM3",
    "ME1 BM4",
    "SMALL HiBM",
]


def make_synthetic_raw_data(n_months: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create compact Fama-French-like raw CSV panels in percent units."""

    rng = np.random.default_rng(1234)
    periods = pd.period_range("2000-01", periods=n_months, freq="M")
    dates = periods.strftime("%Y%m")

    factors_dec = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.006, 0.035, n_months),
            "SMB": rng.normal(0.001, 0.020, n_months),
            "HML": rng.normal(0.002, 0.018, n_months),
            "RF": np.full(n_months, 0.002),
        }
    )
    ff = factors_dec.mul(100.0)
    ff.insert(0, "", dates)

    returns = pd.DataFrame({"": dates})
    for idx, name in enumerate(PORTFOLIOS):
        beta = np.array([0.85 + 0.08 * idx, -0.15 + 0.05 * idx, 0.10 * idx])
        alpha = 0.0005 * (2 - idx)
        shortfall = 0.0004 * idx
        noise = rng.normal(0.0, 0.01 + 0.001 * idx, n_months)
        excess = alpha + factors_dec[["Mkt-RF", "SMB", "HML"]].to_numpy() @ beta
        returns[name] = (factors_dec["RF"] + excess + noise - shortfall) * 100.0

    # Add non-monthly/footer rows to verify the loader selects only the first
    # contiguous monthly panel.
    ff = pd.concat(
        [
            ff,
            pd.DataFrame([["", np.nan, np.nan, np.nan, np.nan]], columns=ff.columns),
            pd.DataFrame([["2020", 1.0, 2.0, 3.0, 0.1]], columns=ff.columns),
        ],
        ignore_index=True,
    )
    returns = pd.concat(
        [
            returns,
            pd.DataFrame(
                [["", *([np.nan] * len(PORTFOLIOS))]], columns=returns.columns
            ),
            pd.DataFrame(
                [
                    [
                        "Average Value Weighted Returns -- Annual",
                        *([np.nan] * len(PORTFOLIOS)),
                    ]
                ],
                columns=returns.columns,
            ),
        ],
        ignore_index=True,
    )
    return returns, ff


@pytest.fixture()
def synthetic_csv_files(tmp_path: Path) -> tuple[Path, Path]:
    ports, factors = make_synthetic_raw_data()
    port_file = tmp_path / "mock_portfolios.csv"
    factor_file = tmp_path / "mock_factors.csv"
    ports.to_csv(port_file, index=False)
    factors.to_csv(factor_file, index=False)
    return port_file, factor_file


@pytest.fixture()
def synthetic_dataset(synthetic_csv_files: tuple[Path, Path]):
    port_file, factor_file = synthetic_csv_files
    return build_factor_dataset(port_file, factor_file, factor_model="ff3")
