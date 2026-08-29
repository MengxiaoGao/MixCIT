from mixci._compat import *  # noqa: F401,F403


# %% CELL 5: Neighborhood construction and kernel gathering
# =============================================================================
# ============================================================================
# Extension of MixCIDebiased (Part 2 methods)
# ============================================================================

def _build_neighborhood(self, Z, X, n, k, z_disc_cols, z_cont_cols,
                        x_disc, fine):
    """
    Build a k_n-NN neighborhood using the composite metric:

      Step 1: Stratify by discrete components of (Z) for coarse, or
              (Z, X) for fine.
      Step 2: Within each stratum, take k_n nearest neighbors using
              the Euclidean distance on standardized continuous
              components (continuous parts of Z for coarse; continuous
              parts of (Z, X) for fine).

    Parameters
    ----------
    Z : (n, d_Z) float64
    X : (n,) float64 -- assumed univariate in this implementation.
        Multivariate X can be supported by passing it column-wise.
    n : int
    k : int
        Target k_n.
    z_disc_cols, z_cont_cols : list of ints
        Indices of discrete and continuous columns of Z.
    x_disc : bool
        Whether X is discrete.
    fine : bool
        True for fine neighborhood (uses (Z, X)); False for coarse
        (uses Z only).

    Returns
    -------
    neighbor_idx : list of np.ndarray
        neighbor_idx[i] = array of neighbor indices for anchor i,
        excluding i itself.
    U_data : (total_pairs, D_cont) float64
        Continuous-coordinate offsets, stacked.
    U_indptr : (n + 1,) int64
        Index pointers into U_data.
    rho : (n,) float64
        Per-anchor k_n-NN radius (the distance to the k-th neighbor).
    """
    # 1. Decide which columns enter the continuous part
    cont_features = []
    if fine and (not x_disc):
        cont_features.append(X.reshape(-1, 1))
    if z_cont_cols:
        cont_features.append(Z[:, z_cont_cols])
    if cont_features:
        cont = np.column_stack(cont_features).astype(np.float64)
        # Standardize per column
        mean = cont.mean(axis=0)
        std = cont.std(axis=0) + 1e-9
        cont_std = (cont - mean) / std
    else:
        cont_std = None

    # 2. Decide which columns enter the discrete-stratification key
    disc_features = []
    if z_disc_cols:
        disc_features.append(Z[:, z_disc_cols])
    if fine and x_disc:
        disc_features.append(X.reshape(-1, 1))
    if disc_features:
        disc = np.column_stack(disc_features)
        # Build group key
        row_keys = np.array(['_'.join(r) for r in disc.astype(str)])
        _, group_ids = np.unique(row_keys, return_inverse=True)
        groups = [np.where(group_ids == g)[0]
                  for g in range(group_ids.max() + 1)]
    else:
        groups = [np.arange(n)]

    # 3. Within each stratum, do k-NN on continuous features
    neighbor_idx = [None] * n
    rho = np.zeros(n, dtype=np.float64)

    for idx in groups:
        m = len(idx)
        if m <= 1:
            for i in idx:
                neighbor_idx[i] = np.array([], dtype=np.int64)
                rho[i] = 0.0
            continue
        if cont_std is None:
            # Pure discrete stratification: every other member of the
            # stratum is a "neighbor", and rho is conventionally 0.
            for i in idx:
                neighbor_idx[i] = idx[idx != i].astype(np.int64)
                rho[i] = 0.0
            continue

        k_sub = min(k, m - 1)
        sub_cont = cont_std[idx]
        nn = NearestNeighbors(n_neighbors=k_sub + 1, algorithm='ball_tree')
        nn.fit(sub_cont)
        dists, lis = nn.kneighbors(sub_cont)
        for loc, i_g in enumerate(idx):
            # Skip the self-neighbor at column 0
            sel = lis[loc, 1:]
            neighbor_idx[i_g] = idx[sel].astype(np.int64)
            rho[i_g] = dists[loc, -1]  # k-th NN distance after self

    # 4. Build flat offset arrays
    if cont_std is not None:
        D_cont = cont_std.shape[1]
        # Total pairs (sum of |neighbor_idx[i]|)
        counts = np.array([len(neighbor_idx[i]) for i in range(n)],
                          dtype=np.int64)
        U_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=U_indptr[1:])
        total = int(U_indptr[-1])
        U_data = np.zeros((total, D_cont), dtype=np.float64)
        for i in range(n):
            nbrs = neighbor_idx[i]
            if len(nbrs) > 0:
                U_data[U_indptr[i]:U_indptr[i + 1], :] = \
                    cont_std[nbrs] - cont_std[i]
    else:
        D_cont = 0
        U_indptr = np.zeros(n + 1, dtype=np.int64)
        U_data = np.zeros((0, 0), dtype=np.float64)

    return neighbor_idx, U_data, U_indptr, rho


def _gather_kernel(self, K_full, neighbor_idx, n):
    """
    Extract kernel values K(Y_i, Y_j) for j in neighbor_idx[i],
    flattened into the same CSR layout as the offsets.

    Returns
    -------
    K_data : (total_pairs,) float64
    K_indptr : (n + 1,) int64
    """
    counts = np.array([len(neighbor_idx[i]) for i in range(n)],
                      dtype=np.int64)
    K_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=K_indptr[1:])
    total = int(K_indptr[-1])
    K_data = np.zeros(total, dtype=np.float64)
    for i in range(n):
        nbrs = neighbor_idx[i]
        if len(nbrs) > 0:
            K_data[K_indptr[i]:K_indptr[i + 1]] = K_full[i, nbrs]
    return K_data, K_indptr