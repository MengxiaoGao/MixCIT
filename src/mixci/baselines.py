from mixci._compat import *  # noqa: F401,F403

from mixci.core import MixCIDebiased  # noqa: F401


# %% CELL 11: Runner factory, baselines, METHODS registry
# =============================================================================
def _debiased_runner_factory(use_permutation=False, n_perms=100):
    """
    Build a runner for the new debiased test.
    """
    def runner(X, Y, Z):
        #from MixCI_debiased import MixCIDebiased
        #import MixCI_debiased_part2  # noqa: F401  patches the class
        #import MixCI_debiased_part3  # noqa: F401  patches the class

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Z is None or (hasattr(Z, 'size') and Z.size == 0):
            data = np.column_stack([X, Y])
            z_indices = []
        else:
            Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
            if Z.shape[0] != len(X):
                Z = Z.T
            data = np.column_stack([X, Y, Z])
            z_indices = list(range(2, data.shape[1]))

        cit = MixCIDebiased(data, k_factor=1.0, B_perm=50, gmb_sims=300,
                            poly_order='auto')
        if use_permutation:
            res = cit.run_ci_test(0, 1, z_indices, return_full=True,
                                  use_permutation_fallback=True,
                                  n_permutations=n_perms)
            return float(res['p_value_perm'])
        return float(cit(0, 1, z_indices))
    return runner


# ============================================================================
# RAW kn-NN DIFFERENCE (baseline; no debiasing, same neighborhoods)
# ============================================================================

def _build_knn_simple(X, Z, k, x_disc, z_disc_cols, z_cont_cols):
    """
    Build fine and coarse kn-NN neighborhoods (composite metric).
    """
    n = len(X)

    cont_F = []
    if not x_disc:
        cont_F.append(X.reshape(-1, 1))
    if z_cont_cols:
        cont_F.append(Z[:, z_cont_cols])
    if cont_F:
        CF = np.column_stack(cont_F).astype(np.float64)
        CF = (CF - CF.mean(axis=0)) / (CF.std(axis=0) + 1e-9)
    else:
        CF = None

    cont_C = []
    if z_cont_cols:
        cont_C.append(Z[:, z_cont_cols])
    if cont_C:
        CC = np.column_stack(cont_C).astype(np.float64)
        CC = (CC - CC.mean(axis=0)) / (CC.std(axis=0) + 1e-9)
    else:
        CC = None

    if z_disc_cols and x_disc:
        df_keys_F = np.array([
            '_'.join(np.append(r.astype(str), str(xi)))
            for r, xi in zip(Z[:, z_disc_cols], X)
        ])
    elif z_disc_cols:
        df_keys_F = np.array(['_'.join(r) for r in Z[:, z_disc_cols].astype(str)])
    elif x_disc:
        df_keys_F = X.astype(str)
    else:
        df_keys_F = None
    if z_disc_cols:
        df_keys_C = np.array(['_'.join(r) for r in Z[:, z_disc_cols].astype(str)])
    else:
        df_keys_C = None

    def build(disc_keys, cont):
        if disc_keys is not None:
            _, gid = np.unique(disc_keys, return_inverse=True)
            groups = [np.where(gid == g)[0] for g in range(gid.max() + 1)]
        else:
            groups = [np.arange(n)]
        nbrs = [np.array([], dtype=np.int64) for _ in range(n)]
        for idx in groups:
            m = len(idx)
            if m <= 1:
                continue
            if cont is None:
                for i in idx:
                    nbrs[i] = idx[idx != i].astype(np.int64)
                continue
            k_sub = min(k, m - 1)
            nn = NearestNeighbors(n_neighbors=k_sub + 1, algorithm='ball_tree')
            nn.fit(cont[idx])
            _, lis = nn.kneighbors(cont[idx])
            for loc, ig in enumerate(idx):
                nbrs[ig] = idx[lis[loc, 1:]].astype(np.int64)
        return nbrs

    return build(df_keys_F, CF), build(df_keys_C, CC)


def run_raw_knn(X, Y, Z, k_factor=1.0, n_perms=50, threshold=10):
    """
    Raw kn-NN-difference test (no debiasing) calibrated by stratified
    permutation. Provides a direct comparison to the debiased version
    using identical neighborhood structures.
    """
    n = len(Y)
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if Z is None or (hasattr(Z, 'size') and Z.size == 0):
        Z = np.zeros((n, 0))
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    if Z.shape[0] != n:
        Z = Z.T

    x_disc = (len(np.unique(X)) < threshold)
    y_disc = (len(np.unique(Y)) < threshold)
    z_disc_cols = [j for j in range(Z.shape[1])
                    if len(np.unique(Z[:, j])) < threshold]
    z_cont_cols = [j for j in range(Z.shape[1]) if j not in z_disc_cols]

    if y_disc:
        Kfull = (Y[:, None] == Y[None, :]).astype(np.float64)
    else:
        d = np.abs(Y[:, None] - Y[None, :])
        positive_d = d[d > 0]
        med = np.median(positive_d) if positive_d.size > 0 else 1.0
        gamma = 1.0 / (2.0 * med * med + 1e-12)
        Kfull = np.exp(-gamma * d * d)

    k_n = int(max(5, min(round(np.sqrt(n) * k_factor), n - 1)))
    fine, coarse = _build_knn_simple(X, Z, k_n, x_disc,
                                      z_disc_cols, z_cont_cols)

    def stat(perm_idx=None):
        K = Kfull if perm_idx is None else Kfull[np.ix_(perm_idx, perm_idx)]
        total = 0.0
        for i in range(n):
            fi, ci = fine[i], coarse[i]
            aF = K[i, fi].mean() if len(fi) > 0 else 0.0
            aC = K[i, ci].mean() if len(ci) > 0 else 0.0
            total += aF - aC
        return total / n

    obs = stat()

    if z_disc_cols:
        keys = np.array(['_'.join(r) for r in Z[:, z_disc_cols].astype(str)])
        _, strata = np.unique(keys, return_inverse=True)
    elif z_cont_cols:
        from sklearn.cluster import KMeans
        n_clusters = min(max(2, n // 20), 50)
        strata = KMeans(n_clusters=n_clusters, n_init=3,
                         random_state=42).fit(Z[:, z_cont_cols]).labels_
    else:
        strata = np.zeros(n, dtype=np.int64)

    null = np.empty(n_perms, dtype=np.float64)
    for b in range(n_perms):
        perm = np.arange(n)
        for s in np.unique(strata):
            idx_s = np.where(strata == s)[0]
            if len(idx_s) > 1:
                perm[idx_s] = np.random.permutation(idx_s)
        null[b] = stat(perm_idx=perm)

    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) + 1e-12
    z = (obs - mu) / sd
    return float(stats.norm.sf(z))


# ============================================================================
# KCI: Kernel Conditional Independence test (HSIC-based, simplified)
# ============================================================================

def _rbf_kernel(X, gamma=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    D2 = np.maximum(D2, 0.0)
    if gamma is None:
        positive = D2[D2 > 0]
        med = np.median(np.sqrt(positive)) if positive.size > 0 else 1.0
        gamma = 1.0 / (2.0 * med * med + 1e-12)
    return np.exp(-gamma * D2)


def run_kci(X, Y, Z, n_perms=100, eps=1e-3):
    """
    Conditional HSIC of (X | Z, Y | Z) using kernel-ridge residuals,
    permutation-calibrated.
    """
    n = len(X)
    X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    if Z is None or Z.size == 0:
        Z = np.zeros((n, 0))
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    if Z.shape[0] != n:
        Z = Z.T

    H = np.eye(n) - np.ones((n, n)) / n
    Kx = H @ _rbf_kernel(X) @ H
    Ky = H @ _rbf_kernel(Y) @ H
    if Z.shape[1] > 0:
        Kz = H @ _rbf_kernel(Z) @ H
        Rzx = eps * np.linalg.solve(Kz + eps * np.eye(n), np.eye(n))
        Kx_res = Rzx @ Kx @ Rzx
        Ky_res = Rzx @ Ky @ Rzx
    else:
        Kx_res = Kx
        Ky_res = Ky

    obs = float(np.trace(Kx_res @ Ky_res)) / (n * n)
    null = np.empty(n_perms, dtype=np.float64)
    for b in range(n_perms):
        p = np.random.permutation(n)
        null[b] = float(np.trace(Kx_res @ Ky_res[np.ix_(p, p)])) / (n * n)
    return float(np.mean(null >= obs))


# ============================================================================
# Fisher's Z: linear partial correlation
# ============================================================================

def run_fisherz(X, Y, Z):
    n = len(X)
    X = np.asarray(X, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    if Z is None or Z.size == 0:
        Z = np.zeros((n, 0))
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    if Z.shape[0] != n:
        Z = Z.T

    if Z.shape[1] == 0:
        rho = np.corrcoef(X, Y)[0, 1]
        df = n - 2
    else:
        Z1 = np.column_stack([np.ones(n), Z])
        Bx, *_ = np.linalg.lstsq(Z1, X, rcond=None)
        By, *_ = np.linalg.lstsq(Z1, Y, rcond=None)
        Xr = X - Z1 @ Bx
        Yr = Y - Z1 @ By
        rho = np.corrcoef(Xr, Yr)[0, 1]
        df = n - Z.shape[1] - 2

    rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(max(df - 1, 1))
    return float(2 * (1 - stats.norm.cdf(np.abs(z))))


# ============================================================================
# CMI-KNN (Frenzel-Pompe / Kraskov)
# ============================================================================

def run_cmiknn(X, Y, Z, k=5, n_perms=50):
    n = len(X)
    X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    if Z is None or Z.size == 0:
        Z = np.zeros((n, 0))
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    if Z.shape[0] != n:
        Z = Z.T

    if Z.shape[1] == 0:
        xyz = np.hstack([X, Y])
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=k + 1, p=np.inf)
        eps = np.maximum(dists[:, k], 1e-12)
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)
        nx = np.array([len(tree_x.query_ball_point(X[i], eps[i], p=np.inf)) - 1
                       for i in range(n)])
        ny = np.array([len(tree_y.query_ball_point(Y[i], eps[i], p=np.inf)) - 1
                       for i in range(n)])
        obs = (digamma(k) + digamma(n)
                - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    else:
        def cmi(x, y, z):
            xyz = np.hstack([x, y, z])
            xz = np.hstack([x, z])
            yz = np.hstack([y, z])
            tree = cKDTree(xyz)
            dists, _ = tree.query(xyz, k=k + 1, p=np.inf)
            eps = np.maximum(dists[:, k], 1e-12)
            tx = cKDTree(xz)
            ty = cKDTree(yz)
            tz = cKDTree(z)
            nxz = np.array([len(tx.query_ball_point(xz[i], eps[i], p=np.inf)) - 1
                            for i in range(n)])
            nyz = np.array([len(ty.query_ball_point(yz[i], eps[i], p=np.inf)) - 1
                            for i in range(n)])
            nz = np.array([len(tz.query_ball_point(z[i], eps[i], p=np.inf)) - 1
                           for i in range(n)])
            return (digamma(k) - np.mean(digamma(nxz + 1))
                    - np.mean(digamma(nyz + 1)) + np.mean(digamma(nz + 1)))
        obs = cmi(X, Y, Z)

    null = np.empty(n_perms, dtype=np.float64)
    for b in range(n_perms):
        Xp = np.random.permutation(X)
        if Z.shape[1] == 0:
            xyp = np.hstack([Xp, Y])
            tree = cKDTree(xyp)
            dists, _ = tree.query(xyp, k=k + 1, p=np.inf)
            eps = np.maximum(dists[:, k], 1e-12)
            tree_x = cKDTree(Xp)
            tree_y = cKDTree(Y)
            nx = np.array([len(tree_x.query_ball_point(Xp[i], eps[i], p=np.inf)) - 1
                           for i in range(n)])
            ny = np.array([len(tree_y.query_ball_point(Y[i], eps[i], p=np.inf)) - 1
                           for i in range(n)])
            null[b] = (digamma(k) + digamma(n)
                       - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
        else:
            null[b] = cmi(Xp, Y, Z)
    return float(np.mean(null >= obs))


# ============================================================================
# Chi-square and G-test (discrete only)
# ============================================================================

def _build_contingency(X, Y, Z):
    n = len(X)
    if Z is None or Z.size == 0:
        Z_keys = np.zeros(n, dtype=np.int64)
    else:
        Z = np.atleast_2d(Z)
        if Z.shape[0] != n:
            Z = Z.T
        if Z.shape[1] == 0:
            Z_keys = np.zeros(n, dtype=np.int64)
        else:
            row_keys = np.array(['_'.join(r) for r in Z.astype(str)])
            _, Z_keys = np.unique(row_keys, return_inverse=True)

    _, X_keys = np.unique(X, return_inverse=True)
    _, Y_keys = np.unique(Y, return_inverse=True)

    L = Z_keys.max() + 1
    nx = X_keys.max() + 1
    ny = Y_keys.max() + 1

    table = np.zeros((L, nx, ny), dtype=np.int64)
    for i in range(n):
        table[Z_keys[i], X_keys[i], Y_keys[i]] += 1
    return table


def run_chisq(X, Y, Z):
    table = _build_contingency(X, Y, Z)
    L, _, _ = table.shape
    chi2_total = 0.0
    df_total = 0
    for l in range(L):
        sub = table[l]
        n_sub = sub.sum()
        if n_sub < 5:
            continue
        rs = sub.sum(axis=1, keepdims=True)
        cs = sub.sum(axis=0, keepdims=True)
        exp = rs @ cs / n_sub
        mask = exp > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_l = np.where(mask, (sub - exp) ** 2 / exp, 0.0).sum()
        df_l = max((sub.shape[0] - 1) * (sub.shape[1] - 1), 1)
        chi2_total += chi2_l
        df_total += df_l
    if df_total == 0:
        return 1.0
    return float(1 - stats.chi2.cdf(chi2_total, df_total))


def run_gsq(X, Y, Z):
    table = _build_contingency(X, Y, Z)
    L, _, _ = table.shape
    g_total = 0.0
    df_total = 0
    for l in range(L):
        sub = table[l]
        n_sub = sub.sum()
        if n_sub < 5:
            continue
        rs = sub.sum(axis=1, keepdims=True)
        cs = sub.sum(axis=0, keepdims=True)
        exp = rs @ cs / n_sub
        mask = (sub > 0) & (exp > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            g_l = 2 * np.where(mask, sub * np.log(sub / exp), 0.0).sum()
        df_l = max((sub.shape[0] - 1) * (sub.shape[1] - 1), 1)
        g_total += g_l
        df_total += df_l
    if df_total == 0:
        return 1.0
    return float(1 - stats.chi2.cdf(g_total, df_total))


# ============================================================================
# Method registry
# ============================================================================

METHODS = {
    'ours_debiased': _debiased_runner_factory(use_permutation=False),
    'ours_debiased_perm': _debiased_runner_factory(
        use_permutation=True, n_perms=100
    ),
    'ours_raw_knn': run_raw_knn,
    'kci': run_kci,
    'fisherz': run_fisherz,
    'cmiknn': run_cmiknn,
    'chisq': run_chisq,
    'gsq': run_gsq,
}


def baselines_for_regime(regime):
    base = ['ours_debiased', 'ours_raw_knn']
    if regime == 'CCC':
        return base + ['kci', 'fisherz', 'cmiknn']
    if regime == 'DDD':
        return base + ['chisq', 'gsq']
    if regime in ('DCC', 'CCD'):
        return base + ['kci', 'cmiknn']
    return base