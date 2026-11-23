# Finite-horizon residual ranking for adaptive McLachlan reduction

Numerical code, archived results, and supplementary figures accompanying the **20-page consolidated research manuscript**, *Finite-Horizon Residual Ranking for Adaptive McLachlan Reduction of Non-Normal Transport–Diffusion Dynamics* (23 November 2026 working revision).

## Scope and limitations

This repository contains reproducibility material for a **noiseless, classical numerical study** of finite-element transport–diffusion systems, finite-horizon transient-energy ranking, and adaptive McLachlan tangent-space reduction. The resolved low-rank examples use a classically constructed diffusion-modal basis. The small gate-product test simulates parameterized Givens factors on a classical computer; it is **not** a quantum hardware demonstration. There is no claim of quantum speedup or systematic superiority of weighted selection.

The favorable `n_h=32` and `n_h=64` eight-direction experiments involve **initial ranking**: all seven additions happen at `t=0`. The trigger was checked using both method-specific and common Euclidean residual thresholds; selected generators and errors were unchanged in these cases. Other benchmarks preserve unfavorable and mixed outcomes.

## Repository layout

- `code/` — primary simulations and independent checking scripts (Python)
- `results/` — saved JSON/CSV outputs and compressed NumPy trajectories
- `figures/` — supporting plots in PDF format
- `supplementary/` — reproducibility notes and reference audit
- `requirements.txt` — Python dependencies
- `SHA256SUMS.txt` — file-integrity hashes for the archived scientific files

## Quick verification

With Python 3, in the repository root:

```bash
python -m pip install -r requirements.txt
python code/verify_tables.py
```

Expected output: `PASS: 7 primary claims match saved numerical results`.

## Recreate numerical analyses

Run from the repository root (the scripts locate `results/` relative to their own files):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/modal_lowrank.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/gate_product_demo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/trigger_cost_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/score_timing.py
```

The scripts can overwrite archived output files, so work in a separate checkout or back up `results/` before regenerating them. The archived timing values are platform-dependent and are not quantum measurement costs. The original full numerical validation script and mesh resolution checks are included for inspection and replication.

## Versioning and citation

This repository records the numerical materials from the 23 November 2026 consolidated revision. For a stable scholarly citation, use a **specific commit SHA** or a separately archived DOI-bearing release (when one exists); the moving `main` branch is not immutable.

No external datasets or private credentials are needed for the archived scripts. See `supplementary/REPRODUCIBILITY.md` for limitations and a script-to-result map.
