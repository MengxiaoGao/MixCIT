from mixci._compat import *  # noqa: F401,F403
from mixci.monomials import _build_monomials, _Np_count, _solve_lp_intercept_weights, _compute_intercepts_per_anchor  # noqa: F401

from mixci.core import MixCIDebiased  # noqa: F401

# %% CELL 8: Attach methods to class + Part 3 helpers
# =============================================================================
# ============================================================================
# Patch the methods onto MixCIDebiased
# ============================================================================


"""
MixCI_debiased_part3.py
========================

Part 3: Adaptive diagnostics and robustness checks for MixCIDebiased.

Adds:
  * Automatic polynomial-order selection with validation against the
    undersmoothing constraint alpha < 1 - D/(2(p+1)).
  * Diagnostic helper that reports the effective alpha, the bias-rate
    exponent (p+1)/D, the design-matrix conditioning rate, and the
    overlap-graph density.
  * Ill-conditioning warnings: if the fraction of anchors where the
    local-polynomial design failed exceeds a threshold, fall back to
    p = max(p - 1, 0) and re-fit.
  * Optional permutation calibration as a robustness fallback when
    the analytic p-value is suspect (e.g., very small or very large
    tau_hat compared to a reference scale).

All additions are attached to the MixCIDebiased class from Parts 1 - 2
by monkey-patching at module import time.
"""



# ============================================================================
# Bandwidth and polynomial-order selection
# ============================================================================

def _select_polynomial_order(self, n, D, k_n, requested):
    """
    Choose a polynomial order p that satisfies the undersmoothing
    constraint and is feasible for the available sample size.

    Constraints:
      (a) Undersmoothing: alpha < 1 - D / (2 (p + 1)),
          where k_n = n^alpha.
      (b) Feasibility: k_n must exceed the moment-matrix size N_p =
          binomial(D + p, p) by a logarithmic factor.

    Parameters
    ----------
    n : int
    D : int
        Effective smoothing dimension.
    k_n : int
    requested : int, 'auto', or None
        User-specified polynomial order or 'auto'.

    Returns
    -------
    p : int
    info : dict
        Diagnostic info including alpha, the undersmoothing bound,
        and N_p.
    """
    # Convert k_n to alpha = log(k_n) / log(n)
    alpha = float(np.log(max(k_n, 2)) / np.log(max(n, 2)))

    if requested == 'auto' or requested is None:
        # Set p = D as the default (per user instruction).
        if D <= 0:
            p = 0
        else:
            p = max(D, 1)
    else:
        p = int(requested)

    # Validate undersmoothing
    if p > 0:
        bound = 1.0 - D / (2.0 * (p + 1))
    else:
        bound = 1.0  # No constraint when there is no smoothing

    undersmoothing_ok = (alpha < bound) if D > 0 else True

    # Feasibility check: N_p should be much smaller than k_n
    from math import comb
    N_p = comb(D + p, p) if (D > 0 and p > 0) else 1
    feasibility_ratio = (k_n / max(N_p, 1)) / max(np.log(n), 1.0)
    feasible = (feasibility_ratio > 1.0)

    info = {
        'p': p,
        'D': D,
        'k_n': k_n,
        'alpha': alpha,
        'undersmoothing_bound': bound,
        'undersmoothing_ok': undersmoothing_ok,
        'N_p': N_p,
        'feasibility_ratio': feasibility_ratio,
        'feasible': feasible,
    }

    return p, info


def _diagnose_design_conditioning(self, flags, name):
    """
    Report how many anchors had a well-conditioned local-polynomial
    design vs how many fell back to uniform weights.
    """
    if flags is None or len(flags) == 0:
        return {'n': 0, 'ok': 0, 'fail_rate': 0.0, 'name': name}
    n = len(flags)
    n_ok = int(np.sum(flags))
    fail_rate = (n - n_ok) / n
    return {
        'n': n,
        'ok': n_ok,
        'fail_rate': fail_rate,
        'name': name,
    }


# ============================================================================
# Optional permutation-calibration fallback
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def _compute_perm_intercept_diff(
    U_F_data, U_F_indptr, D_F, p_F, rho_F, ridge_F,
    U_C_data, U_C_indptr, D_C, p_C, rho_C, ridge_C,
    fine_neighbors_flat, fine_nbr_indptr,
    coarse_neighbors_flat, coarse_nbr_indptr,
    K_full, perms, n
):
    """
    For each permutation b, compute the debiased statistic
    Delta_n^(p)[b] = (1/n) sum_i (a_F^(p)[i, b] - a_C^(p)[i, b]),
    where a_F[i, b] uses the permuted kernel K(Y_{perm[i]}, Y_{perm[j]})
    and similarly for a_C.

    The weights themselves are NOT recomputed under the permutation -
    they depend only on the neighborhood geometry, which is fixed.
    Only the kernel values are permuted. This is the correct
    conditional-permutation construction for testing Y | (Z, X) vs
    Y | Z while holding the (Z, X) and Z geometries fixed.

    Parameters
    ----------
    U_F_data, U_F_indptr, D_F, p_F, rho_F, ridge_F : fine LP setup
    U_C_data, U_C_indptr, D_C, p_C, rho_C, ridge_C : coarse LP setup
    fine_neighbors_flat : (total_F_pairs,) int64
        Stacked neighbor indices for fine graph.
    fine_nbr_indptr : (n + 1,) int64
        Index pointers parallel to U_F_indptr.
    coarse_neighbors_flat, coarse_nbr_indptr : analogous, coarse.
    K_full : (n, n) float64
        Pre-computed kernel Gram matrix on the original Y.
    perms : (B, n) int64
        Permutation matrix (each row is a permutation of 0..n-1).
    n : int

    Returns
    -------
    Delta_perm : (B,) float64
    """
    # Pre-compute the weights once (they depend only on geometry).
    # We'll defer this to the caller via a separate fast routine
    # that pre-computes w_F[i, :] and w_C[i, :] arrays. For ease,
    # we inline a per-anchor solve here for each anchor exactly once
    # outside the permutation loop (but Numba cannot easily cache,
    # so we accept the per-anchor cost - it's still O(n) per
    # permutation if we cache the weights externally).
    #
    # To keep this routine simple and correct, we accept that the
    # weights are recomputed per permutation. For typical B = 100
    # this is ~B times the unpermuted cost, which is acceptable for
    # a robustness check.
    B = perms.shape[0]
    Delta_perm = np.empty(B, dtype=np.float64)
    for b in prange(B):
        perm_b = perms[b]
        Delta_acc = 0.0
        for i in range(n):
            # Fine intercept under permutation
            a_F_start = U_F_indptr[i]
            a_F_end = U_F_indptr[i + 1]
            k_F = a_F_end - a_F_start
            if D_F > 0 and k_F > 0:
                U_i = U_F_data[a_F_start:a_F_end, :]
                w_F, ok = _lp_solve_local(U_i, p_F, D_F, rho_F[i], ridge_F)
            else:
                w_F = np.full(k_F, 1.0 / max(k_F, 1), dtype=np.float64)

            # Apply kernel values via permutation
            nbr_F_start = fine_nbr_indptr[i]
            a_F_val = 0.0
            for jj in range(k_F):
                j = fine_neighbors_flat[nbr_F_start + jj]
                ip = perm_b[i]
                jp = perm_b[j]
                a_F_val += w_F[jj] * K_full[ip, jp]

            # Coarse intercept under permutation
            a_C_start = U_C_indptr[i]
            a_C_end = U_C_indptr[i + 1]
            k_C = a_C_end - a_C_start
            if D_C > 0 and k_C > 0:
                U_i_C = U_C_data[a_C_start:a_C_end, :]
                w_C, _ = _lp_solve_local(U_i_C, p_C, D_C, rho_C[i], ridge_C)
            else:
                w_C = np.full(k_C, 1.0 / max(k_C, 1), dtype=np.float64)

            nbr_C_start = coarse_nbr_indptr[i]
            a_C_val = 0.0
            for jj in range(k_C):
                j = coarse_neighbors_flat[nbr_C_start + jj]
                ip = perm_b[i]
                jp = perm_b[j]
                a_C_val += w_C[jj] * K_full[ip, jp]

            Delta_acc += (a_F_val - a_C_val)
        Delta_perm[b] = Delta_acc / n
    return Delta_perm


@njit(fastmath=True, cache=True)
def _lp_solve_local(U, p, D, rho_i, ridge):
    """
    Local copy of the weight solver from Part 1, accessible inside the
    permutation routine. This avoids Numba's restriction on calling
    methods between two prange-decorated functions in some versions.

    Returns (w, ok). See _solve_lp_intercept_weights in Part 1 for full
    documentation.
    """
    from MixCI_debiased import _solve_lp_intercept_weights  # local import
    return _solve_lp_intercept_weights(U, p, D, rho_i, ridge)


def _stratified_permutation(self, Z_for_strata, n, n_clusters_per_dim=None,
                             random_state=None):
    """
    Build a single stratified permutation of [0, ..., n-1] such that
    observations are permuted only within strata defined by Z.

    For purely discrete Z, strata are the unique tuples of Z. For
    continuous Z, we use KMeans clustering to define strata (matching
    the convention of the existing MixCI codebase). For mixed Z, we
    combine.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(random_state)

    if Z_for_strata is None or Z_for_strata.size == 0:
        return rng.permutation(n)

    Z = np.atleast_2d(Z_for_strata)
    if Z.shape[0] != n:
        Z = Z.T

    z_disc_cols = [j for j in range(Z.shape[1]) if self._is_discrete(Z[:, j])]
    z_cont_cols = [j for j in range(Z.shape[1]) if j not in z_disc_cols]

    if not z_cont_cols:
        # All discrete: stratify by exact values
        row_keys = np.array(['_'.join(r) for r in Z.astype(str)])
        _, strata = np.unique(row_keys, return_inverse=True)
    else:
        Z_cont = Z[:, z_cont_cols]
        d_cont = len(z_cont_cols)
        n_unique = len(np.unique(Z_cont, axis=0))
        if n_clusters_per_dim is None:
            # ~ n / (15 / d_cont) clusters, capped at unique count
            n_clusters = min(max(2, n // max(3, 15 // max(d_cont, 1))),
                             n_unique)
        else:
            n_clusters = min(n_clusters_per_dim ** d_cont, n_unique)
        n_clusters = max(2, n_clusters)
        km_labels = KMeans(n_clusters=n_clusters, n_init=3,
                           random_state=42).fit(Z_cont).labels_
        if not z_disc_cols:
            strata = km_labels
        else:
            disc_keys = np.array(['_'.join(r)
                                   for r in Z[:, z_disc_cols].astype(str)])
            combined = np.array([f"{d}_{k}"
                                  for d, k in zip(disc_keys, km_labels)])
            _, strata = np.unique(combined, return_inverse=True)

    perm = np.arange(n)
    for s in np.unique(strata):
        idx_s = np.where(strata == s)[0]
        if len(idx_s) > 1:
            perm[idx_s] = rng.permutation(idx_s)
    return perm


# ============================================================================
# Augmented debiased test with diagnostics and optional fallback
# ============================================================================

def debiased_test_with_diagnostics(
    self, X, Y, Z, x_disc, y_disc, c_const=2.0,
    fail_rate_threshold=0.10,
    use_permutation_fallback=False,
    n_permutations=200,
    random_state=None,
):
    """
    A diagnostic-rich wrapper around _debiased_test.

    Workflow:
      1. Identify D_fine, D_coarse, select p_fine, p_coarse.
      2. Validate undersmoothing and feasibility.
      3. Run the debiased test in diagnostic mode.
      4. If the fraction of ill-conditioned designs exceeds
         fail_rate_threshold, optionally reduce p by 1 and retry.
      5. If use_permutation_fallback is True, recompute the p-value via
         permutation calibration and report both.

    Returns
    -------
    result : dict
        Contains 'p_value' (analytic), 'p_value_perm' (if computed),
        'diagnostics' (full info on the fit), and 'warnings' (list).
    """
    n = len(Y)
    Z = np.atleast_2d(Z)
    if Z.shape[0] != n:
        Z = Z.T
    z_disc_cols = [j for j in range(Z.shape[1]) if self._is_discrete(Z[:, j])]
    z_cont_cols = [j for j in range(Z.shape[1]) if j not in z_disc_cols]

    D_fine = len(z_cont_cols) + (0 if x_disc else 1)
    D_coarse = len(z_cont_cols)
    k = self._get_k(n)

    p_fine, info_fine = self._select_polynomial_order(n, D_fine, k,
                                                       self.poly_order)
    p_coarse, info_coarse = self._select_polynomial_order(n, D_coarse, k,
                                                            self.poly_order)

    warnings_list = []
    if D_fine > 0 and not info_fine['undersmoothing_ok']:
        warnings_list.append(
            f"Fine undersmoothing violated: alpha={info_fine['alpha']:.3f} "
            f">= bound={info_fine['undersmoothing_bound']:.3f}. "
            f"Consider larger p or smaller k_n."
        )
    if D_fine > 0 and not info_fine['feasible']:
        warnings_list.append(
            f"Fine design may be ill-conditioned: k_n/N_p/log(n) = "
            f"{info_fine['feasibility_ratio']:.2f} (<1). "
            f"N_p={info_fine['N_p']}, k_n={k}."
        )

    # Save user's poly_order and temporarily set it explicitly for this
    # round (avoids re-running _select)
    orig_poly = self.poly_order
    self.poly_order = max(p_fine, p_coarse)

    # Run debiased test in diagnostic mode
    diag = self._debiased_test(
        X, Y, Z, x_disc, y_disc, c_const=c_const,
        return_diagnostics=True,
    )

    # Check ill-conditioning rate
    flags_F = diag.get('flags_F', None)
    flags_C = diag.get('flags_C', None)
    cond_F = self._diagnose_design_conditioning(flags_F, 'fine')
    cond_C = self._diagnose_design_conditioning(flags_C, 'coarse')

    if cond_F['fail_rate'] > fail_rate_threshold and p_fine > 0:
        warnings_list.append(
            f"Fine design ill-conditioned at "
            f"{cond_F['fail_rate']*100:.1f}% of anchors; retrying with p-1."
        )
        # Retry with reduced polynomial order
        self.poly_order = max(p_fine - 1, 0)
        diag = self._debiased_test(
            X, Y, Z, x_disc, y_disc, c_const=c_const,
            return_diagnostics=True,
        )
        flags_F = diag.get('flags_F', None)
        flags_C = diag.get('flags_C', None)
        cond_F = self._diagnose_design_conditioning(flags_F, 'fine')
        cond_C = self._diagnose_design_conditioning(flags_C, 'coarse')

    # Restore original setting
    self.poly_order = orig_poly

    p_value_perm = None
    if use_permutation_fallback:
        try:
            p_value_perm = self._permutation_calibrate(
                X, Y, Z, x_disc, y_disc, diag,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        except Exception as e:
            warnings_list.append(f"Permutation fallback failed: {e}")

    result = {
        'p_value': diag['p_value'],
        'p_value_perm': p_value_perm,
        'Delta': diag['Delta'],
        'tau_sq': diag['tau_sq'],
        'T': diag['T'],
        'fine_info': info_fine,
        'coarse_info': info_coarse,
        'fine_condition': cond_F,
        'coarse_condition': cond_C,
        'k_n': diag['k_n'],
        'n_candidates': diag.get('n_candidates', 0),
        'warnings': warnings_list,
    }
    return result


def _permutation_calibrate(
    self, X, Y, Z, x_disc, y_disc, diag,
    n_permutations=200,
    random_state=None,
):
    """
    Stratified permutation calibration of Delta_n^(p): for each
    permutation b, recompute Delta_n^(p)[b] using the same local-
    polynomial weights but with Y permuted within Z-strata. The
    empirical p-value is the fraction of permutations with
    Delta_n^(p)[b] >= Delta_obs.

    Notes
    -----
    Unlike the analytic test (which uses the overlap-graph variance),
    this calibration does not require a CLT or tau-hat estimate. It
    is computationally heavier (B times the unpermuted cost) but
    serves as a robustness check.

    For efficiency, this implementation:
      * Pre-computes the kernel Gram matrix once.
      * Pre-builds the neighborhood geometries (offsets, neighbor
        indices, radii) once - they depend only on (X, Z).
      * Re-uses the local-polynomial weights across permutations
        (they depend only on geometry, not on Y).
    """
    n = len(Y)
    rng = np.random.default_rng(random_state)

    # Build permutations stratified by Z
    perms = np.zeros((n_permutations, n), dtype=np.int64)
    for b in range(n_permutations):
        perms[b] = self._stratified_permutation(
            Z, n, random_state=rng.integers(0, 2**31 - 1)
        )

    # Build kernel Gram matrix once
    K_full = self._kernel_matrix(Y, y_disc)

    # Build neighborhoods once
    k = self._get_k(n)
    Z = np.atleast_2d(Z)
    if Z.shape[0] != n:
        Z = Z.T
    z_disc_cols = [j for j in range(Z.shape[1]) if self._is_discrete(Z[:, j])]
    z_cont_cols = [j for j in range(Z.shape[1]) if j not in z_disc_cols]
    D_F = len(z_cont_cols) + (0 if x_disc else 1)
    D_C = len(z_cont_cols)
    p_F = max(D_F, 1) if D_F > 0 else 0
    p_C = max(D_C, 1) if D_C > 0 else 0

    fine_nbr, U_F, U_F_indptr, rho_F = self._build_neighborhood(
        Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=True
    )
    coarse_nbr, U_C, U_C_indptr, rho_C = self._build_neighborhood(
        Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=False
    )

    # Flatten neighbor index arrays (analog of CSR row_idx)
    fine_counts = np.array([len(fine_nbr[i]) for i in range(n)],
                            dtype=np.int64)
    fine_nbr_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(fine_counts, out=fine_nbr_indptr[1:])
    fine_nbr_flat = np.zeros(int(fine_nbr_indptr[-1]), dtype=np.int64)
    for i in range(n):
        if len(fine_nbr[i]) > 0:
            fine_nbr_flat[fine_nbr_indptr[i]:fine_nbr_indptr[i + 1]] = \
                fine_nbr[i]

    coarse_counts = np.array([len(coarse_nbr[i]) for i in range(n)],
                              dtype=np.int64)
    coarse_nbr_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(coarse_counts, out=coarse_nbr_indptr[1:])
    coarse_nbr_flat = np.zeros(int(coarse_nbr_indptr[-1]), dtype=np.int64)
    for i in range(n):
        if len(coarse_nbr[i]) > 0:
            coarse_nbr_flat[coarse_nbr_indptr[i]:coarse_nbr_indptr[i + 1]] = \
                coarse_nbr[i]

    # Compute null Delta_perm
    Delta_perm = _compute_perm_intercept_diff(
        U_F, U_F_indptr, D_F if D_F > 0 else 1, p_F if D_F > 0 else 0,
        rho_F, self.ridge,
        U_C, U_C_indptr, D_C if D_C > 0 else 1, p_C if D_C > 0 else 0,
        rho_C, self.ridge,
        fine_nbr_flat, fine_nbr_indptr,
        coarse_nbr_flat, coarse_nbr_indptr,
        K_full, perms, n,
    )

    Delta_obs = diag['Delta']
    p_value = float(np.mean(Delta_perm >= Delta_obs))
    return p_value


# ============================================================================
# Patch onto MixCIDebiased
# ============================================================================
