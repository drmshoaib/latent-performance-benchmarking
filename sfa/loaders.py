from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_MODELS: dict[str, list[str]] = {
    "capm": ["mkt_rf"],
    "ff3": ["mkt_rf", "smb", "hml"],
    "ff5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
    "ff5_mom": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
}

MISSING_SENTINELS = [-99.99, -999.0, -99.9, -999.99]


@dataclass(frozen=True)
class FactorDataset:
    data: pd.DataFrame
    factor_model: str
    factor_cols: list[str]
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp
    n_observations: int
    n_portfolios: int

    def summary(self) -> dict:
        return {
            "factor_model": self.factor_model,
            "factor_cols": ",".join(self.factor_cols),
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "n_observations": self.n_observations,
            "n_portfolios": self.n_portfolios,
        }


def _normalise_column_name(name: object) -> str:
    return (
        str(name).strip().lower().replace("-", "_").replace(" ", "_").replace("__", "_")
    )


def _normalise_date_text(values: pd.Series) -> pd.Series:
    """Return string YYYYMM/annual-like dates without numeric .0 suffixes."""

    raw = values.astype(str).str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_text = numeric.round().astype("Int64").astype(str)
    return raw.where(numeric.isna(), numeric_text)


def _first_monthly_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the first contiguous YYYYMM block in a Fama-French CSV export.

    The downloaded portfolio file can contain several panels in a single CSV
    (monthly value-weighted, monthly equal-weighted, annual panels, counts,
    average firm size, and footers). The first block is the monthly
    value-weighted return panel used by this project.
    """

    first_col = _normalise_date_text(df.iloc[:, 0])
    is_yyyymm = first_col.str.fullmatch(r"\d{6}")
    positions = np.flatnonzero(is_yyyymm.to_numpy())
    if len(positions) == 0:
        raise ValueError("No YYYYMM monthly data block found.")

    start = int(positions[0])
    end = start
    while end < len(df) and bool(is_yyyymm.iloc[end]):
        end += 1

    out = df.iloc[start:end].copy()
    out.rename(columns={out.columns[0]: "date"}, inplace=True)
    return out


def _parse_yyyymm(values: pd.Series) -> pd.Series:
    text = _normalise_date_text(values)
    if not text.str.fullmatch(r"\d{6}").all():
        bad = text[~text.str.fullmatch(r"\d{6}")].head(3).tolist()
        raise ValueError(f"Invalid YYYYMM values found: {bad}")
    periods = pd.PeriodIndex(text, freq="M")
    return periods.to_timestamp(how="end")


def _to_decimal(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") / 100.0
    return out


def load_ff_factors(ff_file: Path) -> pd.DataFrame:
    """Load the first monthly Fama-French factor panel from a CSV file."""

    ff = _first_monthly_block(pd.read_csv(ff_file))
    ff.columns = [_normalise_column_name(c) for c in ff.columns]
    ff.rename(columns={"mktrf": "mkt_rf"}, inplace=True)
    ff.replace(MISSING_SENTINELS, np.nan, inplace=True)

    ff["date"] = _parse_yyyymm(ff["date"])
    factor_cols = [c for c in ff.columns if c != "date"]
    ff = _to_decimal(ff, factor_cols)
    ff = ff.dropna(subset=["date"]).sort_values("date")
    return ff


def load_portfolios(port_file: Path) -> pd.DataFrame:
    """Load the first monthly portfolio-return panel from a CSV file."""

    ports = _first_monthly_block(pd.read_csv(port_file))
    ports.columns = [str(c).strip() for c in ports.columns]
    ports.replace(MISSING_SENTINELS, np.nan, inplace=True)
    ports["date"] = _parse_yyyymm(ports["date"])

    portfolio_cols = [c for c in ports.columns if c != "date"]
    ports = _to_decimal(ports, portfolio_cols)
    ports_long = (
        ports.melt(id_vars="date", var_name="portfolio", value_name="ret")
        .dropna(subset=["ret"])
        .sort_values(["portfolio", "date"])
        .reset_index(drop=True)
    )
    return ports_long


def available_factor_models(factors: pd.DataFrame) -> list[str]:
    """Return factor models supported by the available factor columns."""

    cols = set(factors.columns)
    return [name for name, required in FACTOR_MODELS.items() if set(required) <= cols]


def factor_columns(factors: pd.DataFrame, factor_model: str) -> list[str]:
    """Validate and return the factor columns required by a model name."""

    key = factor_model.lower()
    if key not in FACTOR_MODELS:
        raise ValueError(
            f"Unknown factor model '{factor_model}'. "
            f"Choose from {sorted(FACTOR_MODELS)}."
        )
    required = FACTOR_MODELS[key]
    missing = [col for col in required if col not in factors.columns]
    if missing:
        available = available_factor_models(factors)
        raise ValueError(
            f"Factor model '{factor_model}' requires missing columns {missing}. "
            f"Available models from this data: {available}."
        )
    return required


def build_factor_dataset(
    port_file: Path,
    ff_file: Path,
    *,
    factor_model: str = "ff3",
) -> FactorDataset:
    """Align portfolio returns with factors and compute excess returns."""

    ports = load_portfolios(port_file)
    ff = load_ff_factors(ff_file)
    cols = factor_columns(ff, factor_model)

    required_ff = ["date", "rf", *cols]
    missing = [col for col in required_ff if col not in ff.columns]
    if missing:
        raise ValueError(f"Factor file missing required columns: {missing}")

    df = ports.merge(ff[required_ff], on="date", how="inner")
    df["excess_return"] = df["ret"] - df["rf"]
    df = (
        df[["date", "portfolio", "ret", "rf", "excess_return", *cols]]
        .dropna()
        .sort_values(["portfolio", "date"])
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No observations remain after factor/portfolio alignment.")

    return FactorDataset(
        data=df,
        factor_model=factor_model.lower(),
        factor_cols=cols,
        sample_start=pd.Timestamp(df["date"].min()),
        sample_end=pd.Timestamp(df["date"].max()),
        n_observations=int(len(df)),
        n_portfolios=int(df["portfolio"].nunique()),
    )


def design_matrix(df: pd.DataFrame, factor_cols: list[str]) -> np.ndarray:
    """Build an intercept-plus-factors design matrix."""

    missing = [col for col in factor_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")
    return np.column_stack([np.ones(len(df)), df[factor_cols].to_numpy(float)])


def build_merged_dataset(
    port_file: Path,
    ff3_file: Path,
    factor_model: str = "ff3",
) -> pd.DataFrame:
    """Backward-compatible wrapper used by existing scripts."""

    return build_factor_dataset(
        port_file,
        ff3_file,
        factor_model=factor_model,
    ).data
