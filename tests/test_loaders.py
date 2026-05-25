from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfa.loaders import (
    build_factor_dataset,
    factor_columns,
    load_ff_factors,
    load_portfolios,
)


def test_data_loading_alignment_and_dates(synthetic_csv_files):
    port_file, factor_file = synthetic_csv_files

    ports = load_portfolios(port_file)
    factors = load_ff_factors(factor_file)
    dataset = build_factor_dataset(port_file, factor_file, factor_model="ff3")

    assert pd.api.types.is_datetime64_any_dtype(ports["date"])
    assert pd.api.types.is_datetime64_any_dtype(factors["date"])
    assert dataset.n_portfolios == 5
    assert dataset.n_observations == 5 * 48
    assert dataset.factor_cols == ["mkt_rf", "smb", "hml"]
    assert dataset.data["date"].min() == factors["date"].min()


def test_excess_return_calculation(synthetic_csv_files):
    port_file, factor_file = synthetic_csv_files
    dataset = build_factor_dataset(port_file, factor_file, factor_model="capm")
    first = dataset.data.iloc[0]

    assert np.isclose(first["excess_return"], first["ret"] - first["rf"])
    assert {"ret", "rf", "excess_return", "mkt_rf"} <= set(dataset.data.columns)


def test_factor_model_selection_and_errors(synthetic_csv_files):
    _, factor_file = synthetic_csv_files
    factors = load_ff_factors(factor_file)

    assert factor_columns(factors, "capm") == ["mkt_rf"]
    assert factor_columns(factors, "ff3") == ["mkt_rf", "smb", "hml"]
    with pytest.raises(ValueError, match="Unknown factor model"):
        factor_columns(factors, "bad_model")
    with pytest.raises(ValueError, match="requires missing columns"):
        factor_columns(factors, "ff5")


def test_missing_required_columns_raise_clear_error(tmp_path):
    ports = pd.DataFrame({"": ["200001", "200002"], "P1": [1.0, 2.0]})
    factors = pd.DataFrame(
        {"": ["200001", "200002"], "Mkt-RF": [1.0, 2.0], "RF": [0.1, 0.1]}
    )
    port_file = tmp_path / "ports.csv"
    factor_file = tmp_path / "factors.csv"
    ports.to_csv(port_file, index=False)
    factors.to_csv(factor_file, index=False)

    with pytest.raises(ValueError, match="requires missing columns"):
        build_factor_dataset(port_file, factor_file, factor_model="ff3")
