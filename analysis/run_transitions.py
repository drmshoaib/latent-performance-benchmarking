# analysis/run_transitions.py
from pathlib import Path
import pandas as pd

from analysis.transitions import compute_quintile_transitions

RESULTS = Path("results")
INFILE = RESULTS / "ae_rolling.csv"
OUTFILE = RESULTS / "quintile_transitions_h1.csv"

df = pd.read_csv(INFILE, parse_dates=["window_end"])

mat = compute_quintile_transitions(
    df,
    value_col="AE_H1_mean",
    horizon=1,
    n_quantiles=5,
)

mat.to_csv(OUTFILE)

print("Quintile transition matrix written to:", OUTFILE.resolve())
print(mat.round(3))
