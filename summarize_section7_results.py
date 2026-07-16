import csv
import json
from pathlib import Path

import numpy as np


RESULTS_PATH = Path("section7_results.csv")
OUTPUT_PATH = Path("results/section7_summary.json")


def load_rows():
    with RESULTS_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_column(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def summary(rows):
    output = {}
    for N in sorted({int(row["N"]) for row in rows}):
        group = [row for row in rows if int(row["N"]) == N]
        adaptive = float_column(group, "solves_hybrid")
        uniform = float_column(group, "solves_uniform")
        reductions = 100.0 * (uniform - adaptive) / uniform
        widths = float_column(group, "total_window_width")
        windows = float_column(group, "num_windows")
        l_exact = float_column(group, "L_exact")
        l_cert = float_column(group, "L_cert")
        l_psd = float_column(group, "L_psd")

        output[str(N)] = {
            "instances": len(group),
            "all_connected": all(row["connected"].strip().lower() == "true" for row in group),
            "norms": {
                "L_estimate_mean": float(np.mean(l_exact)),
                "L_loose_mean": float(np.mean(l_cert)),
                "L_psd_mean": float(np.mean(l_psd)),
                "loose_over_estimate_mean": float(np.mean(l_cert / l_exact)),
                "psd_over_estimate_mean": float(np.mean(l_psd / l_exact)),
            },
            "solve_counts": {
                "uniform_median": float(np.median(uniform)),
                "uniform_mean": float(np.mean(uniform)),
                "uniform_population_sd": float(np.std(uniform)),
                "adaptive_median": float(np.median(adaptive)),
                "adaptive_mean": float(np.mean(adaptive)),
                "adaptive_population_sd": float(np.std(adaptive)),
                "relative_count_change_mean_percent": float(np.mean(reductions)),
                "relative_count_change_population_sd_percent": float(np.std(reductions)),
                "relative_count_change_min_percent": float(np.min(reductions)),
                "relative_count_change_median_percent": float(np.median(reductions)),
                "relative_count_change_max_percent": float(np.max(reductions)),
            },
            "unresolved": {
                "full_path_resolved": int(np.count_nonzero(windows == 0.0)),
                "mean_components": float(np.mean(windows)),
                "mean_width": float(np.mean(widths)),
                "maximum_width": float(np.max(widths)),
            },
        }
    return output


if __name__ == "__main__":
    rows = load_rows()
    report = summary(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
