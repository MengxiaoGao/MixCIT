from mixci._compat import *  # noqa: F401,F403
from mixci.monomials import _build_monomials, _Np_count, _solve_lp_intercept_weights, _compute_intercepts_per_anchor  # noqa: F401
from mixci.variance import _compute_overlap_variance, _compute_overlap_variance_flat  # noqa: F401


# %% CELL 7: Marginal test (empty Z)
# =============================================================================
def _marginal_test(self, X, Y, x_disc, y_disc, c_const=2.0):
    """
    Marginal X _|_ Y. Special case: no Z, no coarse neighborhood
    (or rather, coarse neighborhood is the whole sample, giving
    a_{i, C} = mean K). Effectively this is a kernel-based test of
    marginal independence using the debiased k_n-NN difference.
    """
    n = len(Y)
    k = self._get_k(n)
    z_disc_cols = []
    z_cont_cols = []
    Z = np.zeros((n, 0))

    D_fine = 0 if x_disc else 1
    D_coarse = 0

    if self.poly_order == 'auto':
        p_fine = max(D_fine, 1) if D_fine > 0 else 0
    else:
        p_fine = int(self.poly_order)

    # Fine neighborhood: k_n-NN in X-space (continuous X) or
    # stratum-mean (discrete X).
    fine_nbr, U_F_data, U_F_indptr, rho_F = self._build_neighborhood(
        Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=True
    )

    K_full = self._kernel_matrix(Y, y_disc)
    K_F_data, K_F_indptr = self._gather_kernel(K_full, fine_nbr, n)

    if D_fine > 0:
        a_F, _ = _compute_intercepts_per_anchor(
            U_F_data, U_F_indptr, D_fine, p_fine, D_fine,
            rho_F, self.ridge, K_F_data, K_F_indptr
        )
    else:
        a_F = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cnt = K_F_indptr[i + 1] - K_F_indptr[i]
            if cnt > 0:
                a_F[i] = K_F_data[K_F_indptr[i]:K_F_indptr[i + 1]].mean()

    # Coarse a_C: leave-one-out sample mean of K
    s = K_full.sum(axis=1) - np.diag(K_full)
    a_C = s / (n - 1)

    xi = a_F - a_C
    Delta = float(xi.mean())
    zeta = xi - Delta

    # Variance: overlap graph in X-space
    coarse_R = np.full(n, np.inf, dtype=np.float64)  # coarse = whole sample
    # Candidate pairs from fine neighborhood union with all-pairs in
    # coarse (which is full); restrict to fine only for tractability
    cand_pairs = set()
    for i in range(n):
        for l in fine_nbr[i]:
            if i < l:
                cand_pairs.add((i, int(l)))
            elif i > l:
                cand_pairs.add((int(l), i))

    if len(cand_pairs) == 0:
        tau_sq = float(np.mean(zeta * zeta))
    else:
        cand_arr = np.array(sorted(cand_pairs), dtype=np.int64)
        cand_i = cand_arr[:, 0]
        cand_l = cand_arr[:, 1]
        cand_dist_F = np.full(len(cand_pairs), np.inf, dtype=np.float64)
        cand_dist_C = np.zeros(len(cand_pairs), dtype=np.float64)  # all in same coarse

        if not x_disc:
            Xv = X.astype(np.float64)
            Xs = (Xv - Xv.mean()) / (Xv.std() + 1e-9)
            for idx_pair in range(len(cand_pairs)):
                i = int(cand_i[idx_pair])
                l = int(cand_l[idx_pair])
                cand_dist_F[idx_pair] = abs(Xs[i] - Xs[l])
        else:
            for idx_pair in range(len(cand_pairs)):
                i = int(cand_i[idx_pair])
                l = int(cand_l[idx_pair])
                cand_dist_F[idx_pair] = 0.0 if X[i] == X[l] else np.inf

        tau_sq = _compute_overlap_variance_flat(
            zeta, cand_i, cand_l, cand_dist_F, cand_dist_C,
            rho_F, coarse_R, c_const
        )

    tau_sq = max(tau_sq, 1e-12)
    tau = np.sqrt(tau_sq)
    T = np.sqrt(n) * Delta / tau
    return float(norm.sf(T))