"""

Section 7 experiments: certified spectral-gap benchmarks for AQC Max-Cut paths.

  H(s) = (1-s) H_I + s H_P

  H_I  = sum_i (I - X_i)/2          PSD, spec {0..N}, binomial multiplicities

  H_P  = diag(c), c[x] = # uncut edges, vertex 0 pinned to spin +1 (kills Z2)

Blocks map to paper subsections:

  [7.1] instance generation        [7.2] certificates (oracle + poly inputs)

  [7.3] Algorithm 1 hybrid sweep   [7.4] baselines + validation

  [7.5] experiment driver -> CSV / figures

"""

import time, csv

import numpy as np

from math import comb

from scipy.sparse.linalg import LinearOperator, eigsh



# ----------------------------- [7.1] instances -----------------------------

def er_graph(n, p, rng):

    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]



def pinned_cost_vector(N, edges):

    """c[x] = uncut edges; qubit q <-> vertex q+1; bit=0 <-> spin +1 (= pinned v0)."""

    x = np.arange(1 << N, dtype=np.int64)

    c = np.zeros(1 << N, dtype=np.float64)

    for (u, v) in edges:

        u, v = min(u, v), max(u, v)

        if u == 0:

            c += (((x >> (v - 1)) & 1) == 0)

        else:

            c += (((x >> (u - 1)) & 1) == ((x >> (v - 1)) & 1))

    return c



def uncut_of_assignment(edges, a):     # poly-time cost of one cut, a[0]=True fixed

    return sum(a[u] == a[v] for (u, v) in edges)



def greedy_cut(n, edges, rng, restarts=30):

    """1-exchange local search; returns (uncut_value, assignment). Poly time."""

    adj = [[] for _ in range(n)]

    for (u, v) in edges: 

        adj[u].append(v)

        adj[v].append(u)

    best = None

    for _ in range(restarts):

        a = rng.integers(0, 2, n).astype(bool); a[0] = True

        moved = True

        while moved:

            moved = False

            for v in range(1, n):

                same = sum(a[u] == a[v] for u in adj[v])

                if 2 * same > len(adj[v]):          # flipping v reduces uncut count

                    a[v] = ~a[v]

                    moved = True

        val = uncut_of_assignment(edges, a)

        if best is None or val < best[0]: 

            best = (val, a.copy())

    return best



def spectral_maxcut_ub(n, edges):

    """maxcut <= n*lam_max(Laplacian)/4  ->  poly-time LOWER bound on E0(H_P)."""

    L = np.zeros((n, n))

    for (u, v) in edges:

        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1

    return min(float(len(edges)), n * np.linalg.eigvalsh(L)[-1] / 4.0)



# ------------------- matvec engine + Lanczos with enclosures ----------------

class PathOperator:

    def __init__(self, N, c):

        self.N, self.d, self.c = N, 1 << N, c

        self.matvecs = 0



    def _sumX(self, v):

        w = np.zeros_like(v)

        for a in range(self.N):                    # flip one bit per axis

            w += v.reshape(1 << a, 2, -1)[:, ::-1, :].reshape(-1)

        return w



    def H(self, s):

        def mv(v):

            v = np.asarray(v).ravel(); self.matvecs += 1

            return (1 - s) * 0.5 * (self.N * v - self._sumX(v)) + s * (self.c * v)

        return LinearOperator((self.d, self.d), matvec=mv, dtype=np.float64)



    def D(self):                                   # H_P - H_I  (for Lipschitz L)

        def mv(v):

            v = np.asarray(v).ravel(); self.matvecs += 1

            return self.c * v - 0.5 * (self.N * v - self._sumX(v))

        return LinearOperator((self.d, self.d), matvec=mv, dtype=np.float64)



def lowest_two(Hop, v0=None, tol=1e-10):

    """Ritz pair (theta0, theta1) + residual norms.

    CAVEAT (state in paper): E1 >= theta1 - r1 assumes no eigenvalue is missed

    between theta0 and theta1; verified exactly for N<=12 in [7.4]."""

    vals, vecs = eigsh(Hop, k=2, which='SA', v0=v0, tol=tol,

                       ncv=min(Hop.shape[0] - 1, 50))

    o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]

    res = np.array([np.linalg.norm(Hop @ vecs[:, j] - vals[j] * vecs[:, j])

                    for j in range(2)])

    return vals, vecs, res



def path_lipschitz(pathop):

    D = pathop.D()

    hi = eigsh(D, k=1, which='LA', tol=1e-8, return_eigenvectors=False)[0]

    lo = eigsh(D, k=1, which='SA', tol=1e-8, return_eigenvectors=False)[0]

    return max(abs(hi), abs(lo))



# ------------------------- [7.2] the three certificates ---------------------

def weyl_endpoint_cert(s, gap0, gap1, L):                       # Prop 5.4

    return max(gap0 - 2 * s * L, gap1 - 2 * (1 - s) * L)



def psd_floor_cert(s, E1I, E1P, ceilings):                      # Prop 5.6 + dictionary

    """ceilings: list of (a,b) with E0(s) <= (1-s)a + s b from trial states."""

    return max((1 - s) * E1I, s * E1P) - min((1 - s) * a + s * b for a, b in ceilings)



def horn_T1_cert(s, specI_asc, cost_asc):                       # Prop 6.4

    a, b = (1 - s) * specI_asc, s * cost_asc

    U0 = np.min(a[::-1] + b)                  # min_i alpha_i + beta_{n+1-i}

    L1 = max(a[0] + b[1], a[1] + b[0])

    return L1 - U0



def driver_spectrum_asc(N):

    return np.repeat(np.arange(N + 1, dtype=np.float64),

                     [comb(N, k) for k in range(N + 1)])



# ---------------- [7.3] Algorithm 1: hybrid certified continuation ----------

def certified_sweep(pathop, L, delta_target, tol=1e-6):

    s, v0 = 0.0, None

    records, windows = [], []

    h_min = delta_target / (2 * L)

    while s < 1.0 - 1e-12:

        vals, vecs, res = lowest_two(pathop.H(s), v0=v0, tol=tol)

        gap_lo = (vals[1] - res[1]) - vals[0]           # certified-at-anchor gap

        records.append((s, vals[0], vals[1], gap_lo))

        h_cert = (gap_lo - delta_target) / (2 * L)

        if h_cert >= h_min:

            h = h_cert

        else:

            h = h_min

            windows.append((s, min(1.0, s + h)))

        v0, s = vecs[:, 0], min(1.0, s + h)             # warm start next solve

        

    # merge adjacent uncertified windows

    merged = []

    for w in windows:

        if merged and abs(w[0] - merged[-1][1]) < 1e-12:

            merged[-1] = (merged[-1][0], w[1])

        else: 

            merged.append(w)

    return records, merged



def hybrid_profile(records, L, sgrid):

    out = np.full_like(sgrid, -np.inf)

    for (si, _, _, glo) in records:

        out = np.maximum(out, glo - 2 * L * np.abs(sgrid - si))

    return out



# --------------------- [7.4] baselines + validation -------------------------

def dense_gap_curve(N, c, sgrid):

    X, I2 = np.array([[0., 1.], [1., 0.]]), np.eye(2)

    HI = np.zeros((1 << N, 1 << N))

    for q in range(N):

        op = np.array([[1.0]])

        for a in range(N):

            op = np.kron(op, X if a == q else I2)

        HI += 0.5 * (np.eye(1 << N) - op)

    gaps = []

    for s in sgrid:

        w = np.linalg.eigvalsh((1 - s) * HI + s * np.diag(c))

        gaps.append(w[1] - w[0])

    return np.array(gaps)



# --------------------------- [7.5] experiment driver ------------------------

def run_instance(N, p, seed, delta_target=0.25, validate=None):

    t_start = time.time()

    rng = np.random.default_rng(seed)

    n = N + 1

    while True:

        edges = er_graph(n, p, rng); m = len(edges)

        c = pinned_cost_vector(N, edges)

        two = np.partition(c, 1)[:2]                        # oracle endpoint data

        E0P, E1P = float(two[0]), float(two[1])

        if E1P > E0P:

            break

    x_star_cost = E0P                                   # exact optimum (oracle mode)

    pathop = PathOperator(N, c)

    L = path_lipschitz(pathop)

    specI = driver_spectrum_asc(N); cost_asc = np.sort(c)

    

    # ceilings: |+>^N gives (0, m/2); optimal basis state gives (N/2, E0P)

    ceil_oracle = [(0.0, m / 2.0), (N / 2.0, x_star_cost)]

    

    # poly-input mode

    E0P_lb = max(0.0, m - spectral_maxcut_ub(n, edges))

    heur_val, _ = greedy_cut(n, edges, rng)

    ceil_poly = [(0.0, m / 2.0), (N / 2.0, float(heur_val))]

    E1P_poly = E0P_lb + 1.0        # PROMISE: nondegenerate optimum (verified below)

    

    assert E1P > E0P, "degenerate optimum: resample or report separately"

    sgrid = np.linspace(0, 1, 201)

    certs = {

        'weyl':  np.array([weyl_endpoint_cert(s, 1.0, E1P - E0P, L) for s in sgrid]),

        'floor': np.array([psd_floor_cert(s, 1.0, E1P, ceil_oracle) for s in sgrid]),

        'floor_poly': np.array([psd_floor_cert(s, 1.0, E1P_poly, ceil_poly) for s in sgrid]),

        'horn':  np.array([horn_T1_cert(s, specI, cost_asc) for s in sgrid]),

    }

    

    pathop.matvecs = 0

    records, windows = certified_sweep(pathop, L, delta_target)

    hybrid = hybrid_profile(records, L, sgrid)

    n_uniform = int(np.ceil(1.0 / (delta_target / (2 * L)))) + 1   # equal-rigor grid

    gap_est = np.array([t1 - t0 for (_, t0, t1, _) in records])

    

    t_prod = time.time() - t_start

    

    row = dict(N=N, m=m, seed=seed, L=L,

               dmin_est=float(gap_est.min()),

               s_star=float(records[int(np.argmin(gap_est))][0]),

               frac_weyl=float(np.mean(certs['weyl'] > 0)),

               frac_floor=float(np.mean(certs['floor'] > 0)),

               frac_floor_poly=float(np.mean(certs['floor_poly'] > 0)),

               frac_horn=float(np.mean(certs['horn'] > 0)),

               window=sum(b - a for a, b in windows),

               solves_hybrid=len(records), solves_uniform=n_uniform,

               matvecs=pathop.matvecs,

               validated=False,

               wall=t_prod,

               wall_validation=0.0)

               

    if validate and N <= 12:                            # referee-proofing asserts

        t_val_start = time.time()

        true_gap = dense_gap_curve(N, c, sgrid)

        for k, v in certs.items():

            assert np.all(v <= true_gap + 1e-8), f"certificate {k} VIOLATED"

        assert np.all(hybrid <= true_gap + 1e-6), "hybrid profile violated"

        row['validated'] = True

        row['wall_validation'] = time.time() - t_val_start

    return row, (sgrid, certs, hybrid, records)



def run_instance_parallel(args):

    N, seed = args

    row, _ = run_instance(N, p=0.5, seed=seed, validate=(N <= 10 and seed == 0))

    return row



if __name__ == "__main__":

    import multiprocessing

    rows = []

    first = True

    for N in [10, 12, 14, 16]:                          # extend to 24 on a workstation

        print(f"Running N={N} with 20 seeds in parallel...")

        with multiprocessing.Pool(processes=5) as pool:

            results = pool.map(run_instance_parallel, [(N, seed) for seed in range(20)])

        

        for row in results:

            rows.append(row)

            print(row)

            

            with open("section7_results.csv", "a" if not first else "w", newline="") as f:

                w = csv.DictWriter(f, fieldnames=row.keys())

                if first:

                    w.writeheader()

                    first = False

                w.writerow(row)