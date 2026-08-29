from mixci._compat import *  # noqa: F401,F403

def run_ci_test(self, x_idx, y_idx, z_indices, return_full=False,
                 use_permutation_fallback=False,
                 n_permutations=200,
                 c_const=2.0,
                 random_state=None):
    """
    Run the conditional independence test with full diagnostics.

    Parameters
    ----------
    x_idx, y_idx : int
        Column indices for X and Y in self.data.
    z_indices : iterable of int
        Column indices for Z.
    return_full : bool, default=False
        If True, return the full diagnostic dict; if False, just the
        p-value (as in __call__).
    use_permutation_fallback : bool, default=False
        Compute and report a permutation-based p-value alongside the
        analytic one.
    n_permutations : int, default=200
    c_const : float, default=2.0
        Constant c > 1 in the overlap indicator.
    random_state : int or None

    Returns
    -------
    p_value : float, or dict if return_full
    """
    X = self.data[:, x_idx]
    Y = self.data[:, y_idx]
    z_indices = list(z_indices)

    x_disc = self._is_discrete(X)
    y_disc = self._is_discrete(Y)

    if len(z_indices) == 0:
        if x_disc and y_disc:
            p = self._gmb_conditional_discrete(X, Y,
                                                 np.zeros(len(X)))
        else:
            p = self._marginal_test(X, Y, x_disc, y_disc,
                                     c_const=c_const)
        if return_full:
            return {'p_value': p, 'route': 'marginal'}
        return p

    Z = self.data[:, z_indices]
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    z_all_disc = all(self._is_discrete(Z[:, j]) for j in range(Z.shape[1]))

    if x_disc and z_all_disc:
        if Z.shape[1] == 1:
            Z_key = Z[:, 0]
        else:
            Z_key = np.array([tuple(row) for row in Z])
        p = self._gmb_conditional_discrete(X, Y, Z_key)
        if return_full:
            return {'p_value': p, 'route': 'gmb_discrete'}
        return p

    # Continuous/mixed: full debiased test with diagnostics
    result = self.debiased_test_with_diagnostics(
        X, Y, Z, x_disc, y_disc,
        c_const=c_const,
        use_permutation_fallback=use_permutation_fallback,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    result['route'] = 'debiased'
    if return_full:
        return result
    return result['p_value']
