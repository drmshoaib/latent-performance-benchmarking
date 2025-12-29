from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def parse_yyyymm_to_date(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.Period(str(int(x)), freq="M").to_timestamp(how="end")
    except Exception:
        return pd.NaT


def to_decimal(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def load_ff3(ff3_file: Path) -> pd.DataFrame:
    ff = pd.read_csv(ff3_file)
    ff.rename(columns={ff.columns[0]: "date"}, inplace=True)
    ff.columns = [c.strip().lower() for c in ff.columns]

    ff = ff.rename(columns={"mkt-rf": "mkt_rf", "mktrf": "mkt_rf"})
    ff.replace([-99.99, -999, -99.9], np.nan, inplace=True)

    ff["date"] = ff["date"].apply(parse_yyyymm_to_date)
    ff = ff.dropna(subset=["date"])

    ff = to_decimal(ff, ["mkt_rf", "smb", "hml", "rf"])
    ff = ff[["date", "mkt_rf", "smb", "hml", "rf"]].dropna()

    return ff


def load_portfolios(port_file: Path) -> pd.DataFrame:
    ports = pd.read_csv(port_file)
    ports.rename(columns={ports.columns[0]: "date"}, inplace=True)
    ports.replace([-99.99, -999, -99.9], np.nan, inplace=True)

    ports["date"] = ports["date"].apply(parse_yyyymm_to_date)
    ports = ports.dropna(subset=["date"])

    port_cols = [c for c in ports.columns if c != "date"]
    ports = to_decimal(ports, port_cols)

    ports_long = ports.melt(
        id_vars=["date"],
        var_name="portfolio",
        value_name="ret",
    ).dropna(subset=["ret"])

    return ports_long


def build_merged_dataset(
    port_file: Path,
    ff3_file: Path,
) -> pd.DataFrame:
    ports = load_portfolios(port_file)
    ff = load_ff3(ff3_file)

    df = ports.merge(ff, on="date", how="inner")
    df["excess_return"] = df["ret"] - df["rf"]

    df = df[
        ["date", "portfolio", "excess_return", "mkt_rf", "smb", "hml"]
    ].dropna()

    return df.sort_values(["portfolio", "date"]).reset_index(drop=True)
