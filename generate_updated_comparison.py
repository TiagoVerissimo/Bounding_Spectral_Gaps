import os

import matplotlib.pyplot as plt
import numpy as np

from main import (
    PathOperator,
    certified_sweep,
    er_graph,
    is_connected,
    lowest_two,
    pinned_cost_vector,
)


def sample_instance(N, p, seed):
    rng = np.random.default_rng(seed)
    n_vertices = N + 1

    while True:
        edges = er_graph(n_vertices, p, rng)
        cost = pinned_cost_vector(N, edges)
        two_lowest = np.partition(cost, 1)[:2]
        if is_connected(n_vertices, edges) and two_lowest[1] > two_lowest[0]:
            return edges, cost


def lower_envelope(s_grid, anchors, norm_bound):
    envelope = np.full_like(s_grid, -np.inf)
    for s_anchor, gap_lower_bound in anchors:
        envelope = np.maximum(
            envelope,
            gap_lower_bound - 2.0 * norm_bound * np.abs(s_grid - s_anchor),
        )
    return envelope


def sampled_ritz_gap_curve(path_operator, s_grid):
    gaps = []
    warm_start = None
    for s in s_grid:
        values, vectors, _ = lowest_two(path_operator.H(s), v0=warm_start)
        gaps.append(values[1] - values[0])
        warm_start = vectors[:, 0]
    return np.asarray(gaps)


def generate_graphs():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    N = 10
    seed = 0
    edges, cost = sample_instance(N=N, p=0.5, seed=seed)
    path_operator = PathOperator(N, cost)
    n_vertices = N + 1
    total_weight = sum(edge[2] for edge in edges)
    norm_bound = float(total_weight + n_vertices)
    endpoint_gap = float(np.partition(cost, 1)[1] - np.partition(cost, 1)[0])
    endpoint_anchors = [(0.0, 2.0), (1.0, endpoint_gap)]
    s_grid = np.linspace(0.0, 1.0, 201)

    print("Computing sampled floating-point reference gap curve...")
    reference_gap = sampled_ritz_gap_curve(path_operator, s_grid)

    print("Computing Weyl endpoint envelope...")
    endpoint_bound = lower_envelope(s_grid, endpoint_anchors, norm_bound)

    plt.figure(figsize=(8, 5))
    plt.plot(s_grid, reference_gap, "k-", lw=2.5, label="Sampled Ritz gap")
    plt.plot(s_grid, endpoint_bound, color="#3a86c8", lw=1.8, label="Weyl endpoint envelope")
    plt.axhline(0.0, color="#9ea1a5", lw=0.8)
    plt.xlabel("Interpolation parameter $s$")
    plt.ylabel("Gap / lower envelope")
    plt.ylim(-0.2, 1.25 * reference_gap.max())
    plt.title(f"Endpoint Weyl envelope (N={N}, seed={seed})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_updated_N10_no_anchors.png"), dpi=160)
    plt.close()

    budgets = [5, 15, 25]
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    print("Computing budgeted adaptive Weyl envelopes...")

    for axis, budget in zip(axes, budgets):
        records, _, _ = certified_sweep(
            path_operator, norm_bound, max_anchors=budget
        )
        adaptive_anchors = [(s, gap_lower_bound) for s, _, _, gap_lower_bound in records]
        bound = lower_envelope(
            s_grid, endpoint_anchors + adaptive_anchors, norm_bound
        )

        axis.plot(s_grid, reference_gap, color="#1e1e24", lw=2.5, label="Sampled Ritz gap")
        axis.plot(
            s_grid,
            bound,
            color="#3a86c8",
            lw=1.8,
            label=f"Weyl envelope ({len(records)} sparse eigensolves + exact endpoints)",
        )
        axis.axhline(0.0, color="#9ea1a5", lw=0.8)
        axis.set_title(f"Adaptive sparse-eigensolve budget: {budget}", fontsize=11, fontweight="bold")
        axis.set_ylabel("Gap / lower envelope", fontsize=10)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(loc="upper right", fontsize=9)
        axis.set_ylim(-0.2, reference_gap.max() * 1.15)

    axes[-1].set_xlabel("Interpolation parameter $s$", fontsize=11)
    fig.suptitle(f"Budgeted Weyl envelopes: N={N}, seed={seed}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_updated_N10_anchors.png"), dpi=160)
    plt.close()

    print("Generated Weyl-only comparison figures from the continuation implementation.")


if __name__ == "__main__":
    generate_graphs()
