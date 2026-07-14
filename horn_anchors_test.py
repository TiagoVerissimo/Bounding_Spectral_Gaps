import os
import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from maxcut_gap_benchmark import random_maxcut_instance, anticut_diagonal, driver_matrix, exact_gap_curve, driver_spectrum

def horn_anchored_bound(s_grid, anchors, HI, DP):
    """
    Computes the maximum Horn T_1^n bound over a set of anchors s0.
    For each s0, we compute the full spectrum of H(s0).
    For any s, we write H(s) = H(s0) + (s - s0)(H_P - H_I).
    """
    n = HI.shape[0]
    M = (sp.diags(DP) - HI).toarray()
    specM = np.sort(np.linalg.eigvalsh(M))[::-1]  # non-increasing
    
    # Precompute spectra for all anchors
    anchor_specs = {}
    for s0 in anchors:
        H_s0 = (1.0 - s0) * HI + s0 * sp.diags(DP)
        spec_s0 = np.sort(np.linalg.eigvalsh(H_s0.toarray()))[::-1]  # non-increasing
        anchor_specs[s0] = spec_s0
        
    out = np.full_like(s_grid, -np.inf)
    
    for a, s in enumerate(s_grid):
        for s0 in anchors:
            spec_s0 = anchor_specs[s0]
            ds = s - s0
            if ds >= 0:
                y = ds * specM
            else:
                y = ds * specM[::-1]
            
            L1 = max(spec_s0[-1] + y[-2], spec_s0[-2] + y[-1])
            U0 = np.min(spec_s0 + y[::-1])
            val = L1 - U0
            if val > out[a]:
                out[a] = val
                
    return out

def run_test():
    N = 8
    seed = 42
    rng = np.random.default_rng(seed)
    edges = random_maxcut_instance(N, "erdos", 0.5, True, rng)
    
    s_grid = np.linspace(0.0, 1.0, 201)
    
    DP = anticut_diagonal(N, edges, sector=True)
    HI = driver_matrix(N, sector=True)
    
    # Exact gap
    E0, E1 = exact_gap_curve(s_grid, HI, DP)
    gap = E1 - E0
    
    # Standard Horn (endpoints only)
    horn_std = horn_anchored_bound(s_grid, [0.0, 1.0], HI, DP)
    
    # Horn with 5, 10, 20, 40 anchors
    horn_5 = horn_anchored_bound(s_grid, np.linspace(0, 1, 5), HI, DP)
    horn_10 = horn_anchored_bound(s_grid, np.linspace(0, 1, 10), HI, DP)
    horn_20 = horn_anchored_bound(s_grid, np.linspace(0, 1, 20), HI, DP)
    horn_40 = horn_anchored_bound(s_grid, np.linspace(0, 1, 40), HI, DP)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(s_grid, gap, 'k-', lw=2, label="Exact Gap")
    plt.plot(s_grid, horn_std, '--', label="Horn (Endpoints only)")
    plt.plot(s_grid, horn_5, ':', label="Horn (5 anchors)")
    plt.plot(s_grid, horn_10, '-.', label="Horn (10 anchors)")
    plt.plot(s_grid, horn_20, '-', alpha=0.7, label="Horn (20 anchors)")
    plt.plot(s_grid, horn_40, '-', lw=2, label="Horn (40 anchors)")
    
    plt.axhline(0.0, color='gray', lw=0.8)
    plt.xlabel("s")
    plt.ylabel("Gap Bound")
    plt.title(f"Multi-Anchor Horn T_1^n Bound vs Exact Gap (N={N})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    plot_path = os.path.join(outdir, "horn_anchors_test.png")
    plt.savefig(plot_path, dpi=160)
    plt.close()
    
    print(f"Saved plot to {plot_path}")
    print(f"Exact min gap: {gap.min():.5f}")
    print(f"Std Horn min: {horn_std.min():.5f} (frac > 0: {np.mean(horn_std > 0):.2%})")
    print(f"Horn 5 min:   {horn_5.min():.5f} (frac > 0: {np.mean(horn_5 > 0):.2%})")
    print(f"Horn 10 min:  {horn_10.min():.5f} (frac > 0: {np.mean(horn_10 > 0):.2%})")
    print(f"Horn 20 min:  {horn_20.min():.5f} (frac > 0: {np.mean(horn_20 > 0):.2%})")
    print(f"Horn 40 min:  {horn_40.min():.5f} (frac > 0: {np.mean(horn_40 > 0):.2%})")

if __name__ == "__main__":
    run_test()
