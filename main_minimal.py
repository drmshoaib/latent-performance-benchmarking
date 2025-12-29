from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---- import your existing SFA engines ----
from sfa_halfnormal import HalfNormalSFA
from sfa_truncated import TruncatedNormalSFA

# -----------------------------
# CONFIG
# -----------------------------
PORT_FILE = Path("data/25_size_bm_portfolios.csv")
FF3_FILE = Path("data/ff3_factors.csv")

OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

MIN_MONTHS = 120  # stability rule


# -----------------------------
# HELPERS
# -----------------------------
def parse_yyyymm_to_date(x) -> pd.Timestamp:
    """Robust YYYYMM → month-end Timestamp."""
    if pd.isna(x):
        return pd.NaT
    try:
        p = pd.Period(str(int(x)), freq="M")
        return p.to_timestamp(how="end")
    except Exception:
        return pd.NaT


def to_decimal(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def enforce_first_col_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={df.columns[0]: name}, inplace=True)
    return df


# -----------------------------
# 1) LOAD FACTORS (FF3)
# -----------------------------
ff = pd.read_csv(FF3_FILE)
ff = enforce_first_col_name(ff, "date")

ff.columns = [str(c).strip().lower() for c in ff.columns]

ff = ff.rename(
    columns={
        "mkt-rf": "mkt_rf",
        "mktrf": "mkt_rf",
    }
)

ff.replace([-99.99, -999, -99.9, -999.0], np.nan, inplace=True)

ff["date"] = ff["date"].apply(parse_yyyymm_to_date)
ff = ff.dropna(subset=["date"])

needed_ff = ["date", "mkt_rf", "smb", "hml", "rf"]
missing = [c for c in needed_ff if c not in ff.columns]
if missing:
    raise ValueError(f"FF3 missing columns: {missing}")

ff = to_decimal(ff, ["mkt_rf", "smb", "hml", "rf"])
ff = ff[needed_ff].dropna()


# -----------------------------
# 2) LOAD 25 PORTFOLIOS
# -----------------------------
ports = pd.read_csv(PORT_FILE)
ports = enforce_first_col_name(ports, "date")
ports.columns = [str(c).strip() for c in ports.columns]

ports.replace([-99.99, -999, -99.9, -999.0], np.nan, inplace=True)

ports["date"] = ports["date"].apply(parse_yyyymm_to_date)
ports = ports.dropna(subset=["date"])

port_cols = [c for c in ports.columns if c != "date"]
ports = to_decimal(ports, port_cols)

ports_long = ports.melt(
    id_vars=["date"],
    var_name="portfolio",
    value_name="ret",
).dropna(subset=["ret"])


# -----------------------------
# 3) MERGE + EXCESS RETURNS
# -----------------------------
df = ports_long.merge(ff, on="date", how="inner")
df["excess_return"] = df["ret"] - df["rf"]

df = df[["date", "portfolio", "excess_return", "mkt_rf", "smb", "hml"]].dropna()


# -----------------------------
# 4) FIT SFA PER PORTFOLIO
# -----------------------------
param_rows: list[dict] = []
ae_rows: list[pd.DataFrame] = []

for portfolio, g in df.groupby("portfolio"):
    g = g.sort_values("date")

    if len(g) < MIN_MONTHS:
        continue

    y = g["excess_return"].to_numpy(float)
    X = np.column_stack(
        [
            np.ones(len(g)),
            g["mkt_rf"].to_numpy(float),
            g["smb"].to_numpy(float),
            g["hml"].to_numpy(float),
        ]
    )

    # ---- Half-normal ----
    try:
        hn = HalfNormalSFA(y, X).fit()
        ae_hn = np.asarray(hn.TE, float)
        u_hn = np.asarray(hn.u_hat, float)
        hn_ok = True
    except Exception as e:
        ae_hn = np.full(len(g), np.nan)
        u_hn = np.full(len(g), np.nan)
        hn_ok = False
        hn_err = repr(e)
        hn = None

    # ---- Truncated-normal ----
    try:
        tn = TruncatedNormalSFA(y, X).fit()
        ae_tn = np.asarray(tn.TE, float)
        u_tn = np.asarray(tn.u_hat, float)
        tn_ok = True
    except Exception as e:
        ae_tn = np.full(len(g), np.nan)
        u_tn = np.full(len(g), np.nan)
        tn_ok = False
        tn_err = repr(e)
        tn = None

    ae_h1 = np.nanmean(np.column_stack([ae_hn, ae_tn]), axis=1)

    # ---- FIXED TIME-SERIES EXPORT (NO .values) ----
    out_ts = g[["date"]].copy()
    out_ts["portfolio"] = portfolio
    out_ts["AE_HN"] = ae_hn
    out_ts["AE_TN"] = ae_tn
    out_ts["AE_H1"] = ae_h1
    out_ts["u_hat_HN"] = u_hn
    out_ts["u_hat_TN"] = u_tn

    ae_rows.append(out_ts)

    row = {
        "portfolio": portfolio,
        "n_months": int(len(g)),
        "AE_HN_mean": float(np.nanmean(ae_hn)),
        "AE_TN_mean": float(np.nanmean(ae_tn)),
        "AE_H1_mean": float(np.nanmean(ae_h1)),
    }

    if hn_ok:
        s = hn.summary()
        row.update(
            {
                "HN_converged": bool(s.get("converged", False)),
                "HN_loglik": float(s.get("log_likelihood", np.nan)),
                "HN_sigma_v": float(s.get("sigma_v", np.nan)),
                "HN_sigma_u": float(s.get("sigma_u", np.nan)),
                "HN_beta0": float(s["beta"][0]),
                "HN_beta_mkt": float(s["beta"][1]),
                "HN_beta_smb": float(s["beta"][2]),
                "HN_beta_hml": float(s["beta"][3]),
            }
        )
    else:
        row.update({"HN_converged": False, "HN_error": hn_err})

    if tn_ok:
        s = tn.summary()
        row.update(
            {
                "TN_converged": bool(s.get("converged", False)),
                "TN_loglik": float(s.get("log_likelihood", np.nan)),
                "TN_sigma_v": float(s.get("sigma_v", np.nan)),
                "TN_sigma_u": float(s.get("sigma_u", np.nan)),
                "TN_mu": float(s.get("mu", np.nan)),
                "TN_beta0": float(s["beta"][0]),
                "TN_beta_mkt": float(s["beta"][1]),
                "TN_beta_smb": float(s["beta"][2]),
                "TN_beta_hml": float(s["beta"][3]),
            }
        )
    else:
        row.update({"TN_converged": False, "TN_error": tn_err})

    param_rows.append(row)


# -----------------------------
# 5) WRITE OUTPUTS
# -----------------------------
if not param_rows:
    raise RuntimeError("No portfolios fitted — check input data.")

df_params = pd.DataFrame(param_rows).sort_values("AE_H1_mean", ascending=False)
df_params.to_csv(OUTDIR / "sfa_params.csv", index=False)

df_ae = pd.concat(ae_rows, ignore_index=True)
df_ae.to_csv(OUTDIR / "ae_timeseries.csv", index=False)

df_rank = (
    df_params[["portfolio", "AE_H1_mean", "AE_HN_mean", "AE_TN_mean", "n_months"]]
    .sort_values("AE_H1_mean", ascending=False)
    .reset_index(drop=True)
)
df_rank["rank_AE_H1"] = np.arange(1, len(df_rank) + 1)
df_rank.to_csv(OUTDIR / "ae_rankings.csv", index=False)

print("Done. Outputs written to:", OUTDIR.resolve())
