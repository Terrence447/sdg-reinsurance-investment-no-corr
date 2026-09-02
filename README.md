# Stochastic Differential Reinsurance Game and Investment Problem under No Correlation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the reproducible numerical and sensitivity analysis supporting the manuscript **“Stochastic Differential Reinsurance Game Formulation and Investment Problem under No Correlation.”** It studies a finite-horizon, two-insurer zero-sum stochastic differential game with bounded investment controls and proportional reinsurance.

## Mathematical setting

Company 1 minimises and Company 2 maximises the expected CARA utility of the terminal wealth difference, \(X(T)=X_2(T)-X_1(T)\). With independent underwriting and market Brownian motions, the reduced Hamiltonian separates by player and control.

For the baseline compact sets \(K_i=[0,\overline\pi_i]\), positive risk premia, and \(p_i\in[0,1]\), the saddle controls used by the code are

\[
\pi_1^*(t)=\overline\pi_1,\qquad p_1^*(t)=1,
\]

and

\[
\pi_2^*(t)=\min\!\left\{\overline\pi_2,
\frac{\mu_2-r}{\gamma\sigma_2^2}e^{-r(T-t)}\right\},
\qquad
p_2^*(t)=\min\!\left\{1,
\frac{\lambda_2}{\gamma b_2^2}e^{-r(T-t)}\right\}.
\]

The compact investment set is essential. After division by the negative CARA value function, Company 1 maximises a strictly convex quadratic. Without finite investment bounds its reduced Hamiltonian is unbounded above, so no finite saddle point exists.

## Repository contents

| Path | Purpose |
| --- | --- |
| `analysis/generate_figures.py` | Computes equilibrium controls, switching times, value-function quantities, and all sensitivity figures. |
| `analysis/latest_results.txt` | Baseline numerical results and selected sensitivity outcomes. |
| `figures/` | Output directory created by the analysis script. |
| `requirements.txt` | Minimal Python dependencies. |

## Reproduce the analysis

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/Terrence447/sdg-reinsurance-investment-no-corr.git
   cd sdg-reinsurance-investment-no-corr
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Linux or macOS:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies and run the analysis:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python analysis/generate_figures.py
   ```

The script uses repository-relative paths, requires no Google Colab modules, and writes its plots to `figures/`. To refresh the text results on Linux or macOS, run:

```bash
python analysis/generate_figures.py | tee analysis/latest_results.txt
```

## Sensitivity experiments

The analysis separates Company 1 and Company 2 and examines:

1. CARA risk aversion \(\gamma\);
2. excess return \(\mu_i-r\);
3. market volatility \(\sigma_i\);
4. investment cap \(\overline\pi_i\);
5. reinsurance price \(\lambda_i\); and
6. underwriting volatility \(b_i\).

Each experiment reports both investment and retention paths. Where a projected Company 2 control reaches a boundary, the corresponding deterministic boundary-entry time is also calculated. Under the no-short-selling baseline, Company 1 remains at its upper endpoints; this is an equilibrium result, not an omitted response.

Generated figures are intentionally not versioned. This keeps the repository small and ensures that every displayed result can be recreated from the published code.

## Citation

If this software contributes to academic work, cite the repository using [`CITATION.cff`](CITATION.cff). Do not use the placeholder DOI unless a real archival DOI has been issued.

## Author

**Lekau Terrence Letsoalo**  
Department of Mathematics and Applied Mathematics, School of Mathematical and Computer Sciences, University of Limpopo, South Africa

For research correspondence, please use the contact options shown on the author's GitHub profile.

## Status

This repository supports an academic manuscript under development. Numerical parameter values are illustrative rather than empirically calibrated. The current model assumes mutually independent Brownian drivers; correlated-risk extensions are future work.
