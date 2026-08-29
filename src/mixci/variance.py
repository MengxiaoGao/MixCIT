from mixci._compat import *  # noqa: F401,F403


# %% CELL 4: Overlap-graph variance helpers
# =============================================================================
# ============================================================================
# Numba helpers (Part 2: overlap-graph variance)
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def _compute_overlap_variance(
    zeta, n,
    fine_dist_data, fine_dist_indptr, fine_R,
    coarse_dist_data, coarse_dist_indptr, coarse_R,
    c_const
):
    """
    Compute the overlap-graph variance estimator:

        hat-tau^2 = (1/n) sum_{i, l} omega_{i,l} * zeta[i] * zeta[l],

    where omega_{i, l} = 1 if either
        ||(Z_i, X_i) - (Z_l, X_l)||  <=  c * (R_{i, F} + R_{l, F}),
        ||Z_i - Z_l||                <=  c * (R_{i, C} + R_{l, C}).

    The function exploits the fact that omega_{i,l} = 1 only for nearby
    anchors. We pre-pass candidate (i, l) pairs via the union of the
    fine and coarse neighbor graphs (extended by the c factor). For
    each candidate, we then check the exact distance condition.

    Parameters
    ----------
    zeta : (n,) float64
        Centered local scores zeta_i = xi_i - mean(xi).
    n : int
        Sample size.
    fine_dist_data : (total_fine_pairs,) float64
        Stacked fine-graph candidate distances. Candidates are pairs
        (i, l) such that l is within an extended fine ball around i.
    fine_dist_indptr : (n + 1,) int64
        For each anchor i, the slice fine_dist_data[indptr[i]:indptr[i+1]]
        is its candidate distances to other anchors. We also pass a
        parallel array of candidate indices (fine_neighbor_idx) but here
        we only need the distances since zeta[l] is looked up by
        index from a separate companion array. The candidate-index
        array is passed via the closure construction in the caller.
    fine_R : (n,) float64
        Per-anchor fine k_n-NN radius R_{i, F}.
    coarse_dist_data, coarse_dist_indptr, coarse_R : analogous, coarse.
    c_const : float
        The constant c > 1 in the omega indicator.

    Returns
    -------
    tau_sq : float
        The variance estimator.

    Notes
    -----
    Because Numba does not support passing companion index arrays
    cleanly into a multi-array signature with prange, the actual
    implementation in the caller delegates the matched-index lookup
    via flat candidate-index arrays passed in alongside the distances.
    See _compute_overlap_variance_flat below.
    """
    raise NotImplementedError("Use _compute_overlap_variance_flat instead.")


@njit(fastmath=True, parallel=True, cache=True)
def _compute_overlap_variance_flat(
    zeta,
    cand_i, cand_l, cand_dist_F, cand_dist_C,
    fine_R, coarse_R, c_const
):
    """
    Vectorized over a flat candidate list.

    Parameters
    ----------
    zeta : (n,) float64
        Centered local scores.
    cand_i, cand_l : (M,) int64
        Indices of candidate pairs (i, l). M is the total number of
        candidate pairs (de-duplicated, i <= l for upper triangle).
    cand_dist_F : (M,) float64
        Distance ||(Z_i, X_i) - (Z_l, X_l)||. Use np.inf if not
        computable (e.g., stratum mismatch).
    cand_dist_C : (M,) float64
        Distance ||Z_i - Z_l||. Use np.inf if not computable.
    fine_R, coarse_R : (n,) float64
        Per-anchor k_n-NN radii.
    c_const : float
        c > 1.

    Returns
    -------
    tau_sq : float
        Variance estimator.
    """
    n = zeta.shape[0]
    M = cand_i.shape[0]

    # Accumulate per-pair contributions, using thread-local accumulators
    # via prange-safe pattern: write to per-thread arrays then reduce.
    # Simplest approach is reduction over a parallel sum.
    total = 0.0
    diag = 0.0

    # Diagonal contributions (i == l)
    for i in prange(n):
        diag += zeta[i] * zeta[i]

    # Off-diagonal candidate contributions (each appears once with i < l
    # and is counted twice in the symmetric sum)
    for m in prange(M):
        i = cand_i[m]
        l = cand_l[m]
        if i == l:
            continue
        # Compute omega
        thresh_F = c_const * (fine_R[i] + fine_R[l])
        thresh_C = c_const * (coarse_R[i] + coarse_R[l])
        cond_F = (cand_dist_F[m] <= thresh_F)
        cond_C = (cand_dist_C[m] <= thresh_C)
        if cond_F or cond_C:
            total += 2.0 * zeta[i] * zeta[l]  # times 2 for symmetric pair

    return (diag + total) / n