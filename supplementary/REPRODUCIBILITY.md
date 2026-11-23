# Reproducibility and supplementary notes

## Main experiments

| Analysis | Main script | Saved outputs |
| --- | --- | --- |
| Original non-normal transport–diffusion benchmark and comparisons | `code/VQA_main_validation.py` | `results/startup_diagnostics.csv`, `results/vqa_endpoint_bound_*.csv`, `figures/original_startup.pdf` |
| Resolved modal low-rank test and horizon/budget sensitivity | `code/modal_lowrank.py` | `results/modal_n32_dt002.json`, `results/modal_n64_dt0005.json`, `results/modal_lowrank_results.json`, `results/modal_*trajectories*.npz` |
| Parameterized Givens gate-product test | `code/gate_product_demo.py` | `results/gate_product_results.json`, `results/adaptive_gate_product_results.json` |
| Mesh-resolution checks | `code/resolution_check.py` | `results/mesh_resolution_new.json`, `results/mesh_diagnostics.csv` |
| Common versus method-specific trigger and initial additions | `code/trigger_cost_audit.py` | `results/trigger_cost_audit.json` |
| Quadrature/score operation timing | `code/score_timing.py` | `results/dense_cost_timing.json` |
| Saved result consistency check | `code/verify_tables.py` | Terminal PASS/fail (does not regenerate trajectories) |

## Scientific qualifications

- Results use finite-dimensional discretizations and noiseless state-vector calculations; hardware gate compilation, shot estimates, and quantum advantage are not demonstrated.
- The earlier 16-state, 15-direction test achieves the **full real normalized tangent rank**. Low error in that regime is not evidence of low-rank compression.
- The favorable eight-direction case depends on a diffusion eigenbasis, smooth initial condition, and a star-Givens candidate pool; its basis-construction cost is classical and nontrivial.
- In the eight-direction `n_h=32` and `n_h=64` experiments, all greedy additions occur at initial time. The observed advantage is a conditional *initial ranking* result, not demonstrated online adaptive superiority.
- Other benchmarks include mixed/negative comparisons; do not infer universal improvement.
- The conservative endpoint residual bound concerns **semi-discrete error**, not the full continuum PDE discretization error; sampled quadrature estimates are not automatically certified upper bounds.
- Reruns may differ slightly due to numerical-library version and platform; recorded wall-clock timing is not an architectural or quantum-complexity guarantee.

## Archive provenance

Files copied without changing scientific values from the 20-page VQA revision package dated 23 November 2026. `SHA256SUMS.txt` lists checksums of all included numerical scripts, results and supplementary files. Manuscript text and its PDF are intentionally not distributed here; this is the **code/data/supplementary-material** repository.
