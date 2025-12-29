# Risk-Adjusted Portfolio Benchmarking via Latent Performance Decomposition

This project implements a **risk-adjusted portfolio benchmarking framework** that decomposes realised portfolio returns into transitory noise and a **latent performance shortfall relative to a factor-implied benchmark**.  
The resulting latent performance measure provides a **stable, interpretable alternative to traditional alpha-based evaluation**.

The analysis is applied to long-horizon equity portfolios benchmarked against standard risk factors, with both static and rolling-window results.

## Key Findings
- Latent performance rankings show strong cross-sectional structure across size–value portfolios
- Rank persistence is high (Spearman ≈ 0.85 at 1-month horizon)
- Extreme quintiles are highly stable; middle quintiles exhibit symmetric mobility
- Results are robust to rolling-window estimation

---

## Motivation

Standard portfolio evaluation relies heavily on:
- Regression alphas  
- Average excess returns  
- Residual-based statistics  

While useful, these measures:
- Treat deviations as symmetric noise  
- Do not distinguish persistent underperformance from transitory shocks  
- Can be unstable in long-horizon or rolling analyses  

This project addresses these limitations by **explicitly modelling an upper performance envelope implied by factor exposures**, and measuring **systematic shortfall relative to that envelope**.

---

## Core Idea (Intuition)

For each portfolio and time period:
- Factor exposures define an **achievable benchmark**
- Realised returns may fall below this benchmark due to:
  - Random noise  
  - Persistent structural underperformance  

The framework decomposes returns accordingly and produces a **unit-free efficiency score** in (0, 1] that:
- Is robust to noise  
- Is comparable across portfolios  
- Exhibits strong persistence when genuine structure exists  

---

## Data

- **Portfolios:** 25 value-weighted equity portfolios sorted on Size × Book-to-Market  
- **Frequency:** Monthly  
- **Sample:** July 1926 – November 2025  
- **Risk Factors:** Market excess return, SMB, HML, Risk-free rate  

All data are aligned at month-end and cleaned using strict no-imputation rules.

---

## Methodology Overview

1. **Factor Benchmarking**  
   Portfolio excess returns are benchmarked against a standard linear factor model.

2. **Latent Performance Decomposition**  
   Deviations are decomposed into:
   - A symmetric stochastic component (noise)
   - A non-negative latent shortfall component (systematic underperformance)

3. **Efficiency Measure**  
   The latent shortfall is mapped into a risk-adjusted efficiency score:
   AE = exp(-u)

4. **Rolling-Window Estimation**  
   Re-estimation over rolling windows enables stability, persistence, and transition analysis.

---

## Empirical Outputs

- Static cross-sectional efficiency rankings  
- Rank persistence across rolling windows  
- Quintile transition matrices  
- Mobility metrics (stay / improve / deteriorate probabilities)  

---

## Repository Structure

```
Famafrench/
│
├── data/                 # Raw factor and portfolio data
├── sfa/                  # Latent decomposition engines
├── analysis/             # Analysis modules
├── results/              # Generated CSVs and figures
├── latex_tables/         # Auto-generated LaTeX tables
└── README.md
```

---

## How to Run

From the project root:

```
python -m sfa.run_build_dataset
python -m analysis.run_rolling
python -m analysis.run_persistence
python -m analysis.run_transitions
python -m analysis.run_mobility
python -m analysis.plots
```

All outputs are written to the `results/` directory.

---

## Practical Value

- Goes beyond alpha without rejecting factor models  
- Produces stable, interpretable diagnostics  
- Suitable for portfolio monitoring, strategy diagnostics, and benchmarking  

---

## Intended Use

This is a **technical demonstration project** intended for:
- Quant research portfolios  
- Applied finance roles  
- Strategy diagnostics and benchmarking  

It emphasises **clarity, robustness, and empirical behaviour** rather than formal statistical testing.

---

## Author

**Dr. Muhammad Shoaib**  
email: safridi@gmail.com
Quantitative Research / Data Science
