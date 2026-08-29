from mixci._compat import *  # noqa: F401,F403
from mixci.monomials import _build_monomials, _Np_count, _solve_lp_intercept_weights, _compute_intercepts_per_anchor  # noqa: F401
from mixci.variance import _compute_overlap_variance, _compute_overlap_variance_flat  # noqa: F401


def _debiased_test(self, X, Y, Z, x_disc, y_disc, c_const=2.0,
                   return_diagnostics=False):
    """
    Regime dispatcher for the debiased test.

    Let D_fine = # continuous cols in (Z, X); D_coarse = # continuous
    cols in Z alone.

      * Regime IV (D_fine > 0, D_coarse = 0, X continuous):
            specialized branch below.
      * All other regimes (D_coarse > 0):
            generic branch (byte-identical to the original code).
    """
    n = len(Y)
    k = self._get_k(n)

    Z = np.atleast_2d(Z)
    if Z.shape[0] != n:
        Z = Z.T
    z_disc_cols = [j for j in range(Z.shape[1])
                    if self._is_discrete(Z[:, j])]
    z_cont_cols = [j for j in range(Z.shape[1]) if j not in z_disc_cols]

    D_fine = len(z_cont_cols) + (0 if x_disc else 1)
    D_coarse = len(z_cont_cols)

    if D_fine > 0 and D_coarse == 0 and not x_disc:
        return _debiased_test_regime_IV(
            self, X, Y, Z, y_disc, z_disc_cols, k, c_const,
            return_diagnostics=return_diagnostics,
        )

    return _debiased_test_generic(
        self, X, Y, Z, x_disc, y_disc, c_const,
        z_disc_cols, z_cont_cols, D_fine, D_coarse, k,
        return_diagnostics=return_diagnostics,
    )


# ============================================================================
# Regime IV specialization: X continuous, Z discrete
# ============================================================================


def _debiased_test_regime_IV(self, X, Y, Z, y_disc, z_disc_cols, k, c_const,
                              return_diagnostics=False):
    """
    Specialized path when X is continuous and Z is discrete.

    Delta_n^(p) = (1/n) sum_i (a_F^(p)[i] - a_C[i]), where:
      * a_F^(p)[i] = fine local-polynomial intercept over the k_n
        nearest X-neighbors within stratum {j : Z_j = Z_i};
      * a_C[i]    = leave-one-out mean of K(Y_i, Y_j) over the same
        stratum. (No local polynomial: D_coarse = 0.)

    Since a_C is constant within a Z-stratum, the coarse contribution
    to the score xi_i is absorbed by within-stratum centering. The
    variance estimator uses only fine-graph overlap pairs.
    """
    n = len(Y)
    X = np.asarray(X, dtype=np.float64).ravel()

    # X is univariate continuous in these generators. For multivariate
    # continuous X later, generalize D_fine = X.shape[1].
    D_fine = 1
    if self.poly_order == 'auto':
        p_fine = max(D_fine, 1)
    else:
        p_fine = int(self.poly_order)

    # --- Fine neighborhood: exact match on Z, k_n-NN on standardized X ---
    fine_nbr, U_F_data, U_F_indptr, rho_F = self._build_neighborhood(
        Z, X, n, k, z_disc_cols, [], x_disc=False, fine=True
    )

    # --- Y kernel, gathered along fine neighbor lists ---
    K_full = self._kernel_matrix(Y, y_disc)
    K_F_data, K_F_indptr = self._gather_kernel(K_full, fine_nbr, n)

    # --- Fine local-polynomial intercepts ---
    a_F, flags_F = _compute_intercepts_per_anchor(
        U_F_data, U_F_indptr, D_fine, p_fine, D_fine,
        rho_F, self.ridge, K_F_data, K_F_indptr
    )

    # --- Coarse fit: within-stratum leave-one-out mean of K(Y_i, Y_j) ---
    if z_disc_cols:
        disc_Z = Z[:, z_disc_cols]
        z_keys = np.array(['_'.join(r) for r in disc_Z.astype(str)])
    else:
        z_keys = np.zeros(n, dtype=str)
    _, strata = np.unique(z_keys, return_inverse=True)

    a_C = np.zeros(n, dtype=np.float64)
    for s in np.unique(strata):
        idx_s = np.where(strata == s)[0]
        m = len(idx_s)
        if m <= 1:
            continue
        K_sub = K_full[np.ix_(idx_s, idx_s)]
        row_sums = K_sub.sum(axis=1) - np.diag(K_sub)
        a_C[idx_s] = row_sums / (m - 1)

    # --- Debiased statistic ---
    xi = a_F - a_C
    Delta = float(xi.mean())

    # --- Within-stratum centering for the variance ---
    # In Regime IV, a_C is constant within stratum, so xi_i inherits a
    # stratum-dependent mean under H_0. Global centering would leave
    # that structure; stratum-mean subtraction removes it exactly.
    zeta = xi.copy()
    for s in np.unique(strata):
        idx_s = np.where(strata == s)[0]
        if len(idx_s) > 0:
            zeta[idx_s] = xi[idx_s] - xi[idx_s].mean()

    # --- Overlap-graph variance from fine-graph candidates only ---
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

        Xs = (X - X.mean()) / (X.std() + 1e-9)

        n_pairs = len(cand_pairs)
        cand_dist_F = np.full(n_pairs, np.inf, dtype=np.float64)
        cand_dist_C = np.full(n_pairs, np.inf, dtype=np.float64)

        for m in range(n_pairs):
            i = int(cand_i[m])
            l = int(cand_l[m])
            if z_keys[i] == z_keys[l]:
                cand_dist_F[m] = abs(Xs[i] - Xs[l])
                cand_dist_C[m] = 0.0

        # Same-stratum pairs share the coarse neighborhood; treat their
        # coarse radius as zero so the coarse overlap indicator fires
        # exactly for same-stratum pairs.
        rho_C = np.zeros(n, dtype=np.float64)

        tau_sq = _compute_overlap_variance_flat(
            zeta, cand_i, cand_l, cand_dist_F, cand_dist_C,
            rho_F, rho_C, c_const
        )

    tau_sq = max(tau_sq, 1e-12)
    tau = np.sqrt(tau_sq)
    T = np.sqrt(n) * Delta / tau
    p_value = float(norm.sf(T))

    if return_diagnostics:
        return {
            'p_value': p_value,
            'Delta': Delta,
            'tau_sq': tau_sq,
            'T': T,
            'a_F': a_F,
            'a_C': a_C,
            'rho_F': rho_F,
            'rho_C': None,
            'flags_F': flags_F,
            'flags_C': np.ones(n, dtype=np.bool_),
            'D_fine': D_fine,
            'D_coarse': 0,
            'p_fine': p_fine,
            'p_coarse': 0,
            'k_n': k,
            'n_candidates': int(len(cand_pairs)),
            'route': 'regime_IV',
        }
    return p_value


# ============================================================================
# Generic path (Regimes II and III)
# ============================================================================


def _debiased_test_generic(self, X, Y, Z, x_disc, y_disc, c_const,
                            z_disc_cols, z_cont_cols, D_fine, D_coarse, k,
                            return_diagnostics=False):
    """
    Generic path used when D_coarse > 0 (there is at least one continuous
    column in Z). Handles Regimes II (all-continuous) and III (X discrete,
    Z continuous). Byte-identical to the original _debiased_test body.
    """
    n = len(Y)

    if self.poly_order == 'auto':
        p_fine = max(D_fine, 1) if D_fine > 0 else 0
        p_coarse = max(D_coarse, 1) if D_coarse > 0 else 0
    else:
        p_fine = int(self.poly_order)
        p_coarse = int(self.poly_order)

    fine_nbr, U_F_data, U_F_indptr, rho_F = self._build_neighborhood(
        Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=True
    )
    coarse_nbr, U_C_data, U_C_indptr, rho_C = self._build_neighborhood(
        Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=False
    )

    K_full = self._kernel_matrix(Y, y_disc)
    K_F_data, K_F_indptr = self._gather_kernel(K_full, fine_nbr, n)
    K_C_data, K_C_indptr = self._gather_kernel(K_full, coarse_nbr, n)

    if D_fine > 0 and U_F_data.shape[0] > 0:
        a_F, flags_F = _compute_intercepts_per_anchor(
            U_F_data, U_F_indptr, D_fine, p_fine, D_fine,
            rho_F, self.ridge, K_F_data, K_F_indptr
        )
    else:
        a_F = np.zeros(n, dtype=np.float64)
        flags_F = np.ones(n, dtype=np.bool_)
        for i in range(n):
            cnt = K_F_indptr[i + 1] - K_F_indptr[i]
            if cnt > 0:
                a_F[i] = K_F_data[K_F_indptr[i]:K_F_indptr[i + 1]].mean()

    if D_coarse > 0 and U_C_data.shape[0] > 0:
        a_C, flags_C = _compute_intercepts_per_anchor(
            U_C_data, U_C_indptr, D_coarse, p_coarse, D_coarse,
            rho_C, self.ridge, K_C_data, K_C_indptr
        )
    else:
        a_C = np.zeros(n, dtype=np.float64)
        flags_C = np.ones(n, dtype=np.bool_)
        for i in range(n):
            cnt = K_C_indptr[i + 1] - K_C_indptr[i]
            if cnt > 0:
                a_C[i] = K_C_data[K_C_indptr[i]:K_C_indptr[i + 1]].mean()

    xi = a_F - a_C
    Delta = float(xi.mean())
    zeta = xi - Delta

    cand_pairs = set()
    for i in range(n):
        for l in fine_nbr[i]:
            if i < l:
                cand_pairs.add((i, int(l)))
            elif i > l:
                cand_pairs.add((int(l), i))
        for l in coarse_nbr[i]:
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

        cont_feats_F, cont_feats_C = [], []
        if not x_disc:
            cont_feats_F.append(X.reshape(-1, 1))
        if z_cont_cols:
            cont_feats_F.append(Z[:, z_cont_cols])
            cont_feats_C.append(Z[:, z_cont_cols])

        if cont_feats_F:
            CF = np.column_stack(cont_feats_F).astype(np.float64)
            CF_std = (CF - CF.mean(axis=0)) / (CF.std(axis=0) + 1e-9)
        else:
            CF_std = None

        if cont_feats_C:
            CC = np.column_stack(cont_feats_C).astype(np.float64)
            CC_std = (CC - CC.mean(axis=0)) / (CC.std(axis=0) + 1e-9)
        else:
            CC_std = None

        if z_disc_cols:
            disc_Z = Z[:, z_disc_cols]
            disc_C_keys = np.array(['_'.join(r) for r in disc_Z.astype(str)])
        else:
            disc_C_keys = None

        if z_disc_cols and x_disc:
            disc_F_arr = np.column_stack([Z[:, z_disc_cols], X.reshape(-1, 1)])
            disc_F_keys = np.array(['_'.join(r) for r in disc_F_arr.astype(str)])
        elif z_disc_cols:
            disc_F_keys = disc_C_keys
        elif x_disc:
            disc_F_keys = X.astype(str)
        else:
            disc_F_keys = None

        n_pairs = len(cand_pairs)
        cand_dist_F = np.full(n_pairs, np.inf, dtype=np.float64)
        cand_dist_C = np.full(n_pairs, np.inf, dtype=np.float64)

        for m in range(n_pairs):
            i = int(cand_i[m])
            l = int(cand_l[m])
            if disc_F_keys is None or disc_F_keys[i] == disc_F_keys[l]:
                if CF_std is not None:
                    diff = CF_std[i] - CF_std[l]
                    cand_dist_F[m] = float(np.sqrt(np.dot(diff, diff)))
                else:
                    cand_dist_F[m] = 0.0
            if disc_C_keys is None or disc_C_keys[i] == disc_C_keys[l]:
                if CC_std is not None:
                    diff = CC_std[i] - CC_std[l]
                    cand_dist_C[m] = float(np.sqrt(np.dot(diff, diff)))
                else:
                    cand_dist_C[m] = 0.0

        tau_sq = _compute_overlap_variance_flat(
            zeta, cand_i, cand_l, cand_dist_F, cand_dist_C,
            rho_F, rho_C, c_const
        )

    tau_sq = max(tau_sq, 1e-12)
    tau = np.sqrt(tau_sq)
    T = np.sqrt(n) * Delta / tau
    p_value = float(norm.sf(T))

    if return_diagnostics:
        return {
            'p_value': p_value,
            'Delta': Delta,
            'tau_sq': tau_sq,
            'T': T,
            'a_F': a_F,
            'a_C': a_C,
            'rho_F': rho_F,
            'rho_C': rho_C,
            'flags_F': flags_F,
            'flags_C': flags_C,
            'D_fine': D_fine,
            'D_coarse': D_coarse,
            'p_fine': p_fine,
            'p_coarse': p_coarse,
            'k_n': k,
            'n_candidates': int(len(cand_pairs)),
            'route': 'generic',
        }
    return p_value
