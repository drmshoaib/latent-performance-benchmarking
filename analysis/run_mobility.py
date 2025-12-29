# analysis/run_mobility.py
from pathlib import Path
import pandas as pd

from analysis.mobility import compute_mobility_metrics

RESULTS = Path("results")
INFILE = RESULTS / "quintile_transitions_h1.csv"
OUTFILE = RESULTS / "quintile_mobility_h1.csv"

mat = pd.read_csv(INFILE, index_col=0)

mobility = compute_mobility_metrics(mat)
mobility.to_csv(OUTFILE, index=False)

print("Mobility metrics written to:", OUTFILE.resolve())
print(mobility.round(3))
