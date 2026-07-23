# Bounding Spectral Gaps: Reproducibility Guide

This repository contains the implementation, numerical data, generated images, and verification tools for a study of spectral-gap bounds for adiabatic quantum-computing Hamiltonians.

The computational experiments apply an adaptive continuation method to symmetry-reduced weighted Max-Cut instances. The method estimates local spectral gaps with sparse eigensolvers and propagates those estimates over intervals of the interpolation parameter using a shift-invariant Weyl bound.

The archived study contains 60 instances at effective problem sizes `N=10`, `N=12`, and `N=14`, with 20 seeded graphs at each size. This guide explains how to replay the archived evidence or recompute the complete numerical experiment.

## Archived cohort

The primary Level 2 archive contains exactly **60 instances**:

- effective qubit sizes `N=10`, `N=12`, and `N=14`, corresponding to `n_v=11`, `n_v=13`, and `n_v=15` physical vertices;
- seeds `0` through `19` once at each size;
- connected weighted Erdős--Rényi graphs with `p=0.5` and a pinned endpoint optimum that `verify_artifacts.py` checks exactly for nondegeneracy.

## Scope and caveats

The aggregate CSV and sparse-eigensolver records are conditional numerical results. Ritz residuals are recorded, but the calculations do not independently verify eigenvalue indexing or use enclosure arithmetic. The selected `N=10`, seed-0 dense-grid comparison is a diagnostic consistency check rather than a continuous-path certificate.

The experiments evaluate finite-size behavior, coverage, and sparse-eigensolve workload. They do not establish asymptotic gap scaling, adiabatic runtime bounds, or computational speedup.

Sparse ARPACK results can differ slightly across SciPy, BLAS, CPU, and thread configurations. Graph and artifact identities use exact SHA-256 values; endpoint nondegeneracy uses exact dyadic arithmetic with no floating tolerance; Level 2 floating-point algebra uses explicit numerical tolerances. Wall-clock timings depend on the platform, BLAS, load, process count, and scheduler and are not expected to reproduce exactly.

## Numerical conventions

Archived binary64 edge weights are interpreted as exact dyadic rationals. If `W_exact` is their exact rational sum, then `W_upper` is the least binary64 value satisfying `W_upper >= W_exact`, and

\[
K_{\mathrm{gap,cert}}
=\operatorname{up}_{64}\!\left(
\operatorname{exact}_{64}(W_{\mathrm{upper}})+\Lambda_I
\right),
\qquad
\Lambda_I=2\left\lfloor\frac{n_v}{2}\right\rfloor.
\]

Here `up_64` denotes least upward binary64 rounding. The compatibility column `W` aliases `W_upper`; `W_model` remains the ordinary rounded Python sum used for model diagnostics. These conventions make the archived weight sums, propagation constants, and cross-file checks reproducible without changing the conditional status of the sparse-eigensolver bounds.

## File map

- `main.py`: shared Level 2 implementation; regenerates the 40-row `N=10,12` base cohort.
- `extend_level2_n14.py`: retains the base rows, evaluates seeds 0--19 at `N=14`, and atomically writes the 60-row aggregate CSV, graph archive, and run manifest.
- `section7_results.csv`: result-schema-4 aggregate metrics for the 60 retained instances, including exact rational and upward-rounding metadata.
- `summarize_section7_results.py`: validates the cohort and exact rounding chain, then writes deterministic JSON and TeX-derived summaries.
- `results/section7_summary.json`: deterministic machine-readable aggregate summary.
- `results/section7_summary.tex`: generated table macros and a derived artifact required by verification.
- `plot_new_results.py`: generates aggregate solve-count and endpoint-coverage images; it does not generate a complexity-scaling plot.
- `generate_updated_comparison.py`: generates the two `N=10`, seed-0 Weyl-envelope images from the archived graph record.
- `scratch/plot_grid_vs_gap.py`: generates the `N=10`, `N=12`, and `N=14` seed-0 solve-density images from archived graph records.
- `test_main.py`: exercises the analytic gap-width bound and continuation edge cases, including upward rounding, floor probes, invalid parameters, and zero-diameter paths.
- `verify_artifacts.py`: cross-checks the CSV, run manifest, graph metadata and hashes, summaries, selected Level 2 interval algebra, and required images; it also enumerates every pinned endpoint configuration to prove nondegeneracy for all 60 graph-schema-2 records.
- `maxcut_gap_benchmark.py`: separate exploratory benchmark and shared Hamiltonian library; it does not generate the Section 7 archive.

## Reference environment

The recorded reference run used Python 3.12.10. From the repository root, install the pinned dependencies:

```bash
python --version
python -m pip install -r requirements.txt
```

For the closest numerical reproduction, use the pinned Python dependencies and record the SciPy, BLAS, CPU, and process configuration alongside any recomputed results.

## Replay archived results

Use this route to verify the evidence included in the repository without repeating the expensive sparse-eigensolver experiment. Run, in order, from the repository root:

```bash
python -m unittest -v test_main.py
python summarize_section7_results.py --check
python verify_artifacts.py
```

The unit suite exercises the propagation constant, directed upward rounding, continuation edge cases, floor probing, and the zero-diameter branch. The summary check reconstructs both deterministic summaries in memory and requires byte-for-byte agreement with the archived files. The verifier then checks:

- result-schema-4 rows for `N=10,12,14` and seeds 0--19;
- the run manifest, individual graph records, canonical graph archive, source hashes, and cross-file SHA-256 links;
- exact interpretation of every binary64 edge weight, directed rounding of the propagation bound, and exact endpoint nondegeneracy by enumeration of all pinned configurations;
- the selected `N=10`, seed-0 continuation log and its interval algebra;
- deterministic summary files and provenance metadata embedded in all required images.

The verifier is read-only and returns a nonzero status for an incomplete cohort, stale derived artifact, inconsistent hash, invalid rounding record, endpoint degeneracy, or missing image.

## Complete recomputation

This route replaces the archived numerical outputs. It performs many sparse ARPACK solves and can take several hours. Run the commands in this order:

```bash
python main.py
python extend_level2_n14.py --processes 5
python summarize_section7_results.py
python plot_new_results.py
python generate_updated_comparison.py
python scratch/plot_grid_vs_gap.py
python verify_artifacts.py
```

The order matters:

1. `main.py` regenerates the fixed 40-instance base cohort for `N=10,12`, seeds 0--19. It replaces `section7_results.csv`, `results/graph_instances.jsonl`, and `results/section7_run_manifest.json` with the intermediate 40-row cohort and uses five worker processes.
2. `extend_level2_n14.py` retains those rows, computes the 20 `N=14` instances, and atomically replaces the aggregate CSV, graph archive, and manifest with the final 60-row cohort. Set `--processes` to a smaller positive number if memory is limited; this changes concurrency, not the requested instances.
3. `summarize_section7_results.py` validates the final cohort and regenerates `results/section7_summary.json` and `results/section7_summary.tex`.
4. The three plotting commands regenerate the checked images from the validated CSV and archived graph records.
5. `verify_artifacts.py` checks the complete regenerated dependency chain.

Do not run summary, plotting, or verification steps between steps 1 and 2: the intermediate archive intentionally contains only 40 rows, while the validators require all 60.

## Artifact mapping

`results/section7_summary.tex` is generated solely from `section7_results.csv` and contains these macros:

| Generated macro | Source fields |
| --- | --- |
| `\SectionSevenGapDiameterRows` | estimated spectral diameter and `K_gap_cert` |
| `\SectionSevenBoundComparisonRows` | previous, uncentered, and shift-invariant propagation constants |
| `\SectionSevenCoverageRows` | 201-point endpoint-bound coverage fractions |
| `\SectionSevenSolveCountRows` | uniform-grid and adaptive sparse-eigensolve counts |
| `\SectionSevenGlobalConditionalRows` | per-run global conditional lower bounds |
| `\SectionSevenUnresolvedRows` | merged unresolved-window counts and widths |

Generated image outputs map to commands as follows:

| Command | Outputs |
| --- | --- |
| `python generate_updated_comparison.py` | `results/comparison_updated_N10_no_anchors.png`, `results/comparison_updated_N10_anchors.png` |
| `python plot_new_results.py` | `results/efficiency_comparison.png`, `results/certificate_coverage.png` |
| `python scratch/plot_grid_vs_gap.py` | `results/grid_density_vs_gap.png`, `results/grid_density_vs_gap_N12.png`, `results/grid_density_vs_gap_N14.png` |

All plotting scripts use a headless Matplotlib backend, resolve paths relative to the repository root, and embed the source CSV hash in PNG metadata. Instance-specific plots also embed the corresponding graph-record hash.

## Numerical expectations

The deterministic summaries derive the `N=12` seed-9 and seed-13 call counts, the single width-`0.001` unresolved window, the `N=14` seed-7 maximum, ensemble means and medians, and resolved-run counts from the same 60 CSV rows. The integral-cost check is `56.52` estimated calls versus 57 observed calls for the selected `N=10`, seed-0 row and `results/conditional_log_N10_seed0.json` record. `verify_artifacts.py` cross-checks that record against the CSV and graph data.

Graph generation is fixed by the archived seeds and graph schema. Exact identities, rational metadata, and endpoint enumeration must match; documented floating-point algebra must remain within the verifier's tolerances. Sparse-eigensolver values may vary slightly, and wall times are diagnostic rather than exact reproducibility targets.

## Exploratory benchmark

The standalone exploratory driver has a separate output contract:

```bash
python maxcut_gap_benchmark.py -h
```

Its outputs are not checked as part of the archived Section 7 artifact chain.

## License

Copyright (c) 2026 Tiago Verissimo. This repository is distributed under the [MIT License](LICENSE). Reuse is permitted subject to the license notice and disclaimer; users remain responsible for checking compatibility with third-party dependencies.
