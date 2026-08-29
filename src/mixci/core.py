from mixci._compat import *  # noqa: F401,F403

# %% CELL 3: MixCIDebiased class (constructor, dispatch, GMB path, stubs)
# =============================================================================
# ============================================================================
# Public class
# ============================================================================

class MixCIDebiased:
    """
    Conditional independence test based on the local-polynomial debiased
    statistic Delta_n^(p) for continuous/mixed regimes, and on the raw
    Delta_n with a Gaussian multiplier bootstrap for the fully-discrete
    regime.

    Parameters
    ----------
    data : array-like of shape (n, d_total)
        Sample matrix with all variables in columns.
    k_factor : float, default=1.0
        Constant in k_n = k_factor * sqrt(n).
    B_perm : int, default=50
        Number of permutation/bootstrap replicates for robustness checks.
    gmb_sims : int, default=300
        Number of Gaussian multiplier bootstrap samples (discrete regime).
    poly_order : int or 'auto', default='auto'
        Local-polynomial order p. 'auto' sets p = D, where D is the
        smoothing dimension of the fine neighborhood.
    ridge : float, default=1e-6
        Ridge added to the local-polynomial design matrix.
    discrete_threshold : int, default=10
        A column with fewer than this many unique values is treated as
        discrete.
    """

    def __init__(self, data, k_factor=1.0, B_perm=50, gmb_sims=300,
                 poly_order='auto', ridge=1e-6, discrete_threshold=10):
        self.data = np.asarray(data, dtype=np.float64)
        self.k_factor = k_factor
        self.B_perm = B_perm
        self.gmb_sims = gmb_sims
        self.poly_order = poly_order
        self.ridge = ridge
        self.discrete_threshold = discrete_threshold

    # ----------------------------------------------------------------
    # Type detection and helpers
    # ----------------------------------------------------------------

    def _is_discrete(self, arr):
        return len(np.unique(arr)) < self.discrete_threshold

    def _get_k(self, n):
        return int(max(5, min(np.round(np.sqrt(n) * self.k_factor), n - 1)))

    def _gaussian_gamma(self, Y):
        """Median-heuristic bandwidth for the Gaussian RBF on Y."""
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
        d = pdist(Y, metric='euclidean')
        med = np.median(d)
        if med < 1e-9:
            med = np.std(Y) if np.std(Y) > 1e-9 else 1.0
        return 1.0 / (2.0 * med * med)

    def _kernel_matrix(self, Y, y_disc):
        """Full Gram matrix K(Y_i, Y_j). Use indicator kernel if discrete."""
        Y = np.asarray(Y).ravel()
        if y_disc:
            return (Y[:, None] == Y[None, :]).astype(np.float64)
        gamma = self._gaussian_gamma(Y)
        dY = Y[:, None] - Y[None, :]
        return np.exp(-gamma * dY * dY)

    # ----------------------------------------------------------------
    # Public dispatch
    # ----------------------------------------------------------------

    def __call__(self, x_idx, y_idx, z_indices):
        """
        Run the CI test for X_x_idx vs Y_y_idx given Z_z_indices.
        Returns a single p-value in [0, 1].
        """
        X = self.data[:, x_idx]
        Y = self.data[:, y_idx]
        z_indices = list(z_indices)

        x_disc = self._is_discrete(X)
        y_disc = self._is_discrete(Y)

        if len(z_indices) == 0:
            # Marginal X _|_ Y
            return self._marginal_test(X, Y, x_disc, y_disc)

        Z = self.data[:, z_indices]
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        z_all_disc = all(self._is_discrete(Z[:, j]) for j in range(Z.shape[1]))

        if x_disc and z_all_disc:
            # Fully discrete (Z, X) -> GMB on raw Delta_n
            if Z.shape[1] == 1:
                Z_key = Z[:, 0]
            else:
                Z_key = np.array([tuple(row) for row in Z])
            return self._gmb_conditional_discrete(X, Y, Z_key)

        # Otherwise: local-polynomial debiased test
        return self._debiased_test(X, Y, Z, x_disc, y_disc)

    # ----------------------------------------------------------------
    # Discrete-regime: Gaussian multiplier bootstrap on Delta_n
    # ----------------------------------------------------------------

    @staticmethod
    def _map_groups(keys):
        keys_arr = np.asarray(keys)
        if keys_arr.ndim == 1:
            _, ids = np.unique(keys_arr, return_inverse=True)
        else:
            _, ids = np.unique(keys_arr, axis=0, return_inverse=True)
        counts = np.bincount(ids)
        return ids, counts

    @staticmethod
    def _onehot_sparse(Y):
        Y_arr = np.asarray(Y)
        n = Y_arr.shape[0]
        _, y_ids = np.unique(Y_arr, return_inverse=True)
        nc = y_ids.max() + 1
        return sparse.csr_matrix(
            (np.ones(n), (np.arange(n), y_ids)), shape=(n, nc)
        )

    def _gmb_stat(self, Y_oh, P_z, P_zx, w_z, w_zx, weights_vec, xi=None):
        if xi is None:
            cz = P_z @ Y_oh
            czx = P_zx @ Y_oh
            term_z = float(np.dot(
                cz.power(2).sum(axis=1).A.ravel() if sparse.issparse(cz)
                else np.sum(cz ** 2, axis=1), w_z))
            term_zx = float(np.dot(
                czx.power(2).sum(axis=1).A.ravel() if sparse.issparse(czx)
                else np.sum(czx ** 2, axis=1), w_zx))
            diag = float(np.sum(weights_vec))
        else:
            xi = np.asarray(xi)
            Y_w = sparse.diags(xi) @ Y_oh
            cz = P_z @ Y_w
            czx = P_zx @ Y_w
            term_z = float(np.dot(
                cz.power(2).sum(axis=1).A.ravel() if sparse.issparse(cz)
                else np.sum(cz ** 2, axis=1), w_z))
            term_zx = float(np.dot(
                czx.power(2).sum(axis=1).A.ravel() if sparse.issparse(czx)
                else np.sum(czx ** 2, axis=1), w_zx))
            diag = float(np.dot(weights_vec, xi ** 2))
        return term_zx + term_z - diag

    def _gmb_conditional_discrete(self, X, Y, Z_key):
        n = len(X)
        z_ids, z_counts = self._map_groups(Z_key)
        zx_ids, zx_counts = self._map_groups(list(zip(Z_key, X)))

        with np.errstate(divide='ignore'):
            w_z = -1.0 / (z_counts - 1)
            w_zx = 1.0 / (zx_counts - 1)
        w_z[z_counts <= 1] = 0.0
        w_zx[zx_counts <= 1] = 0.0

        weights_vec = w_zx[zx_ids] + w_z[z_ids]

        ones = np.ones(n)
        cols = np.arange(n)
        P_z = sparse.csr_matrix((ones, (z_ids, cols)), shape=(len(w_z), n))
        P_zx = sparse.csr_matrix((ones, (zx_ids, cols)), shape=(len(w_zx), n))

        Y_oh = self._onehot_sparse(Y)
        stat_obs = self._gmb_stat(Y_oh, P_z, P_zx, w_z, w_zx, weights_vec)

        sim_xi = np.random.randn(self.gmb_sims, n)
        sim_stats = np.empty(self.gmb_sims, dtype=np.float64)
        for b in range(self.gmb_sims):
            sim_stats[b] = self._gmb_stat(
                Y_oh, P_z, P_zx, w_z, w_zx, weights_vec, xi=sim_xi[b]
            )
        return float(np.mean(sim_stats >= stat_obs))

    # ----------------------------------------------------------------
    # Continuous/mixed-regime: debiased test (Part 1 stub)
    # ----------------------------------------------------------------

    def _marginal_test(self, X, Y, x_disc, y_disc):
        """Marginal X _|_ Y. Placeholder for Part 2/3; uses unconditional
        debiased statistic via empty-Z fine neighborhood."""
        # For Part 1, defer: just return 1.0 (no rejection) if called.
        # We'll implement this properly when we wire up Part 2.
        return 1.0

    def _debiased_test(self, X, Y, Z, x_disc, y_disc):
        """
        Compute Delta_n^(p) and return a p-value.

        Part 1 implements the statistic computation; Part 2 will plug in
        the studentized statistic with the overlap-graph variance.
        """
        n = len(Y)
        k = self._get_k(n)

        # Determine which (Z, X) components are continuous
        Z = np.atleast_2d(Z)
        if Z.shape[0] != n:
            Z = Z.T
        z_disc_cols = [j for j in range(Z.shape[1]) if self._is_discrete(Z[:, j])]
        z_cont_cols = [j for j in range(Z.shape[1]) if j not in z_disc_cols]

        # Fine neighborhood: continuous components include continuous Z and
        # continuous X. Coarse: continuous components include continuous Z only.
        D_fine = len(z_cont_cols) + (0 if x_disc else 1)
        D_coarse = len(z_cont_cols)

        # Effective polynomial order
        if self.poly_order == 'auto':
            p_fine = max(D_fine, 1)
            p_coarse = max(D_coarse, 1)
        else:
            p_fine = int(self.poly_order)
            p_coarse = int(self.poly_order)

        # Build neighborhoods and offsets
        U_fine_data, U_fine_indptr, rho_fine = self._build_neighborhood(
            Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=True
        )
        U_coarse_data, U_coarse_indptr, rho_coarse = self._build_neighborhood(
            Z, X, n, k, z_disc_cols, z_cont_cols, x_disc, fine=False
        )

        # Compute Y-kernel values along the neighbor lists
        K_full = self._kernel_matrix(Y, y_disc)
        K_fine, _ = self._gather_kernel(K_full, U_fine_indptr, n)
        K_coarse, _ = self._gather_kernel(K_full, U_coarse_indptr, n)

        # Replace this; we need actual neighbor index lists too for kernel lookup.
        # The _build_neighborhood method below returns both offsets and the
        # corresponding neighbor index arrays. We restructure accordingly.
        # (Placeholder - real implementation needs neighbor indices.)
        return 1.0  # filled in by Part 2

    # Helper stubs to be elaborated in subsequent parts ---------------

    def _build_neighborhood(self, Z, X, n, k, z_disc_cols, z_cont_cols,
                            x_disc, fine):
        """
        Build the fine (fine=True) or coarse (fine=False) k_n-NN
        neighborhoods using the composite metric:
          1. Restrict to observations matching on discrete components.
          2. Among those, take k_n nearest neighbors in Euclidean space
             on standardized continuous components.

        Returns (offset_data, indptr, rho_array) along with neighbor
        index arrays - to be expanded in Part 2.
        """
        raise NotImplementedError("Filled in fully in Part 2.")

    def _gather_kernel(self, K_full, indptr, n):
        raise NotImplementedError("Filled in fully in Part 2.")