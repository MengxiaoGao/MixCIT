from mixci._compat import *  # noqa: F401,F403


# %% CELL 2: Numba helpers (design matrix, monomials, per-anchor solver)
# =============================================================================

@njit(fastmath=True, cache=True)
def _build_monomials(u, p, D):
    """
    Vector of monomials of total degree 0..p in u in R^D, evaluated at u.
    Length = N_p = binom(D + p, p). Ordered as: [1, u_1, ..., u_D,
    u_1^2, u_1 u_2, ..., u_D^2, ..., total-degree p].

    Implementation uses a deterministic enumeration of multi-indices to
    keep the order stable across calls.
    """
    # Count
    if D == 0:
        out = np.empty(1, dtype=np.float64)
        out[0] = 1.0
        return out

    # Precount N_p
    Np = 1
    for d in range(p):
        Np = Np * (D + d + 1) // (d + 1)

    out = np.empty(Np, dtype=np.float64)
    out[0] = 1.0
    idx = 1
    # We enumerate multi-indices of total degree 1, 2, ..., p
    # by a recursive-style loop. For efficiency, we hand-roll for small p.
    if p == 0:
        return out
    if p >= 1:
        for d in range(D):
            out[idx] = u[d]
            idx += 1
    if p >= 2:
        for d1 in range(D):
            for d2 in range(d1, D):
                out[idx] = u[d1] * u[d2]
                idx += 1
    if p >= 3:
        for d1 in range(D):
            for d2 in range(d1, D):
                for d3 in range(d2, D):
                    out[idx] = u[d1] * u[d2] * u[d3]
                    idx += 1
    if p >= 4:
        for d1 in range(D):
            for d2 in range(d1, D):
                for d3 in range(d2, D):
                    for d4 in range(d3, D):
                        out[idx] = u[d1] * u[d2] * u[d3] * u[d4]
                        idx += 1
    # For p >= 5 we fall back to a generic monomial generator. Most
    # practical cases (D <= 4 continuous components) do not require it.
    if p >= 5:
        # Generic recursive enumeration via a simple lexicographic walk.
        # We allocate a multi-index buffer and update it. Numba-friendly.
        # For total degree d (5 <= d <= p), enumerate all weakly-increasing
        # sequences of length d in 0..D-1.
        buf = np.empty(p, dtype=np.int64)
        for d in range(5, p + 1):
            # Initialize sequence to [0, 0, ..., 0] of length d
            for i in range(d):
                buf[i] = 0
            done = False
            while not done:
                # Emit monomial: product of u[buf[0..d-1]]
                v = 1.0
                for i in range(d):
                    v *= u[buf[i]]
                out[idx] = v
                idx += 1
                # Increment the weakly-increasing sequence
                # Find the rightmost position we can increment
                pos = d - 1
                while pos >= 0 and buf[pos] == D - 1:
                    pos -= 1
                if pos < 0:
                    done = True
                else:
                    buf[pos] += 1
                    for j in range(pos + 1, d):
                        buf[j] = buf[pos]
    return out


def _Np_count(D, p):
    """Pure-Python count of N_p = binomial(D + p, p)."""
    if D == 0:
        return 1
    Np = 1
    for d in range(p):
        Np = Np * (D + d + 1) // (d + 1)
    return Np


@njit(fastmath=True, cache=True)
def _solve_lp_intercept_weights(U, p, D, rho_i, ridge):
    """
    Solve for the local-polynomial intercept weights at a single anchor.

    Parameters
    ----------
    U : (k, D) ndarray of float64
        Continuous-coordinate offsets U_ij = (positions_j - position_i),
        already restricted to neighbors j of i.
    p : int
        Polynomial order.
    D : int
        Continuous dimension.
    rho_i : float
        Bandwidth at anchor i. U will be rescaled by 1/rho_i internally.
    ridge : float
        Small ridge added to the design matrix for numerical stability.

    Returns
    -------
    w : (k,) ndarray of float64
        Weights w_ij such that the local-polynomial intercept at anchor i
        is sum_j w_ij * K(Y_i, Y_j). The weights satisfy sum_j w_ij = 1
        (the constant-reproduction identity).
    ok : bool
        True if the design matrix was well-conditioned; False if a
        fallback to uniform weights was used.
    """
    k = U.shape[0]
    Np = _build_monomials(np.zeros(D), p, D).shape[0]

    # Build the design matrix M = (1/k) sum_j q(u_j) q(u_j)^T
    # and the vector b = (1/k) sum_j q(u_j) [the "first row of M"].
    M = np.zeros((Np, Np), dtype=np.float64)
    Q = np.empty((k, Np), dtype=np.float64)

    inv_rho = 1.0 / rho_i if rho_i > 1e-12 else 1.0
    for j in range(k):
        # Rescale U[j, :] by 1/rho_i and build q
        u_resc = np.empty(D, dtype=np.float64)
        for d in range(D):
            u_resc[d] = U[j, d] * inv_rho
        q = _build_monomials(u_resc, p, D)
        for a in range(Np):
            Q[j, a] = q[a]
            for b in range(Np):
                M[a, b] += q[a] * q[b]
    for a in range(Np):
        for b in range(Np):
            M[a, b] /= k
        M[a, a] += ridge

    # Solve M^{-1} e_0, where e_0 is the first standard basis vector.
    # The weights are then w_j = (1/k) * (Q[j, :] dot (M^{-1} e_0)).
    # We solve M v = e_0 via numpy's linalg.solve.
    e0 = np.zeros(Np, dtype=np.float64)
    e0[0] = 1.0
    try:
        v = np.linalg.solve(M, e0)
        ok = True
    except Exception:
        # Fallback: uniform weights (recovers p = 0)
        w_unif = np.full(k, 1.0 / k, dtype=np.float64)
        return w_unif, False

    # Sanity: catch ill-conditioning (numpy.linalg.solve may still
    # return a vector even when M is near-singular).
    cond_max = 0.0
    for a in range(Np):
        if abs(v[a]) > cond_max:
            cond_max = abs(v[a])
    if cond_max > 1e8:
        # Fallback to uniform
        w_unif = np.full(k, 1.0 / k, dtype=np.float64)
        return w_unif, False

    w = np.empty(k, dtype=np.float64)
    for j in range(k):
        s = 0.0
        for a in range(Np):
            s += Q[j, a] * v[a]
        w[j] = s / k
    return w, True


@njit(fastmath=True, parallel=True, cache=True)
def _compute_intercepts_per_anchor(
    U_data, U_indptr, U_dim, p, D, rho_array, ridge,
    Y_kernel_data, Y_kernel_indptr
):
    """
    Compute the local-polynomial intercepts a_i for all anchors i.

    Parameters
    ----------
    U_data : (total_pairs, D) float64
        Continuous offsets, stacked across all anchors.
    U_indptr : (n + 1,) int64
        Index pointers: U_data[U_indptr[i]:U_indptr[i+1], :] are the
        offsets for anchor i.
    U_dim : int
        D (continuous dimension); equal to U_data.shape[1]. Passed
        explicitly because numba JIT can't index .shape easily.
    p : int
        Polynomial order.
    D : int
        Effective dimension; equal to U_dim.
    rho_array : (n,) float64
        Per-anchor bandwidth rho_i.
    ridge : float
        Regularization for the design matrix.
    Y_kernel_data : (total_pairs,) float64
        Kernel values K(Y_i, Y_j), stacked in the same order as U_data.
    Y_kernel_indptr : (n + 1,) int64
        Index pointers parallel to U_indptr for kernel values.

    Returns
    -------
    intercepts : (n,) float64
        a_i = sum_j w_ij K(Y_i, Y_j), the local-polynomial intercept.
    flags : (n,) bool
        Whether each anchor's design was well-conditioned.
    """
    n = rho_array.shape[0]
    intercepts = np.zeros(n, dtype=np.float64)
    flags = np.ones(n, dtype=np.bool_)

    for i in prange(n):
        a_start = U_indptr[i]
        a_end = U_indptr[i + 1]
        k_i = a_end - a_start
        if k_i <= 0:
            intercepts[i] = 0.0
            flags[i] = False
            continue

        U_i = U_data[a_start:a_end, :]
        rho_i = rho_array[i]

        # Compute local-polynomial weights for this anchor
        w, ok = _solve_lp_intercept_weights(U_i, p, D, rho_i, ridge)
        flags[i] = ok

        # Weighted sum: a_i = sum_j w_j K(Y_i, Y_j)
        k_start = Y_kernel_indptr[i]
        s = 0.0
        for j in range(k_i):
            s += w[j] * Y_kernel_data[k_start + j]
        intercepts[i] = s

    return intercepts, flags