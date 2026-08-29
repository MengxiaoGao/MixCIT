from mixci._compat import *  # noqa: F401,F403

# %% CELL 10: DataGenerator (all four regimes)
# =============================================================================
class DataGenerator:
    def __init__(self, n_samples=1000, seed=None):
        self.n = n_samples
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def _noise(self, scale=None):
        if scale is None:
            scale = np.random.uniform(0.1, 0.3)
        return np.random.normal(scale=scale, size=self.n)

    def _discretize(self, data, bins=5):
        return pd.qcut(data, q=bins, labels=False, duplicates='drop').astype(int)

    def _random_nonlinear(self, x):
        choice = np.random.randint(0, 4)
        if choice == 0: return np.sin(x)
        elif choice == 1: return np.cos(x)
        elif choice == 2: return np.tanh(x)
        else: return x ** 2 / (1 + x ** 2)

    # ------------------------------------------------------------------
    # CASE 1: CCC (continuous X, Y, Z)
    # ------------------------------------------------------------------
    def generate_ccc(self, mechanism='linear', independent=True):
        # randomize coefficients
        a_xz = np.random.uniform(0.5, 1.5)
        a_yz = np.random.uniform(0.5, 1.5)
        a_yx = np.random.uniform(0.3, 0.8)

        if mechanism == 'linear':
            Z = np.random.uniform(-2, 2, self.n)
            X = a_xz * Z + self._noise()
            if independent:
                Y = a_yz * Z + self._noise()
            else:
                Y = a_yz * Z + a_yx * X + self._noise()

        elif mechanism == 'nonlinear':
            Z = np.random.uniform(-2, 2, self.n)
            X = self._random_nonlinear(a_xz * Z) + self._noise()
            if independent:
                Y = self._random_nonlinear(a_yz * Z) + self._noise()
            else:
                Y = self._random_nonlinear(a_yz * Z) + a_yx * self._random_nonlinear(X) + self._noise()

        elif mechanism == 'neural':
            Z = np.random.uniform(-2, 2, self.n)
            X = self._random_nn_mapping(Z) + self._noise()
            if independent:
                Y = self._random_nn_mapping(Z) + self._noise()
            else:
                Y = self._random_nn_mapping(np.column_stack((Z, X))) + self._noise()

        # ---------- New: Mediator structure ----------
        elif mechanism == 'mediator':
            # X -> Z -> Y, no direct X->Y when independent=True
            X = np.random.normal(0, 1, self.n)
            Z = a_xz * X + self._noise()
            if independent:
                Y = a_yz * Z + self._noise()          # only indirect effect
            else:
                Y = a_yz * Z + a_yx * X + self._noise()  # add direct effect

        # ---------- New: Collider structure ----------
        elif mechanism == 'collider':
            if independent:
                # X, Y, Z all independent
                X = np.random.normal(0, 1, self.n)
                Y = np.random.normal(0, 1, self.n)
                Z = np.random.normal(0, 1, self.n)
            else:
                # X and Y independent, but Z = f(X,Y) + noise (collider)
                X = np.random.normal(0, 1, self.n)
                Y = np.random.normal(0, 1, self.n)
                Z = a_xz * X + a_yz * Y + self._noise()

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        return X, Y, Z

    # ------------------------------------------------------------------
    # CASE 2: CCD (Z discrete, X cont, Y cont)
    # ------------------------------------------------------------------
    def generate_cc_d(self, mechanism='linear', independent=True):
        n_categories = np.random.choice([2, 3, 4])
        a_xz = np.random.uniform(0.5, 1.5)
        a_yz = np.random.uniform(0.5, 1.5)
        a_yx = np.random.uniform(0.3, 0.8)

        if mechanism == 'linear':
            Z = np.random.randint(0, n_categories, self.n)
            Z_effect_X = np.linspace(-2, 2, n_categories)[Z] * a_xz
            Z_effect_Y = np.linspace(2, -2, n_categories)[Z] * a_yz
            X = Z_effect_X + self._noise()
            if independent:
                Y = Z_effect_Y + self._noise()
            else:
                Y = Z_effect_Y + a_yx * X + self._noise()

        elif mechanism == 'nonlinear':
            X = np.random.normal(0, 1, self.n)
            p_z = 1 / (1 + np.exp(-X))
            Z = np.random.binomial(n=1, p=p_z, size=self.n)
            if independent:
                # H0: X _|_ Y | Z
                Y = np.sin(Z * np.pi) + self._noise()
            else:
                # H1
                Y = np.sin(Z * np.pi) + 0.8 * (X**2) + self._noise()



            #Z = np.random.randint(0, n_categories, self.n)
            #Z_effect_X = np.linspace(-2, 2, n_categories)[Z] * a_xz
            #Z_effect_Y = np.linspace(2, -2, n_categories)[Z] * a_yz
            #X = self._random_nonlinear(Z_effect_X) + self._noise()
            #if independent:
            #    Y = self._random_nonlinear(Z_effect_Y) + self._noise()
            #else:
            #    Y = self._random_nonlinear(Z_effect_Y) + a_yx * self._random_nonlinear(X) + self._noise()

        # Mediator (X -> Z -> Y, but Z discrete)
        elif mechanism == 'mediator':
            X = np.random.normal(0, 1, self.n)
            # latent continuous Z* then discretize
            Z_cont = a_xz * X + self._noise()
            Z = self._discretize(Z_cont, bins=n_categories)
            # Effect of Z on Y: use mean of each category
            z_means = np.linspace(-2, 2, n_categories)
            Z_effect_Y = z_means[Z] * a_yz
            if independent:
                Y = Z_effect_Y + self._noise()
            else:
                Y = Z_effect_Y + a_yx * X + self._noise()

        # Collider (X and Y independent, Z discrete collider)
        elif mechanism == 'collider':
            if independent:
                X = np.random.normal(0, 1, self.n)
                Y = np.random.normal(0, 1, self.n)
                Z = np.random.randint(0, n_categories, self.n)
            else:
                X = np.random.normal(0, 1, self.n)
                Y = np.random.normal(0, 1, self.n)
                # Z depends on X and Y through continuous latent then discretize
                Z_cont = a_xz * X + a_yz * Y + self._noise()
                Z = self._discretize(Z_cont, bins=n_categories)

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        return X, Y, Z

    # ------------------------------------------------------------------
    # CASE 3: DCC (Z cont, X discrete, Y cont)
    # ------------------------------------------------------------------
    def generate_dc_c(self, mechanism='linear', independent=True):
        n_bins = np.random.choice([3, 4, 5])
        a_xz = np.random.uniform(0.5, 1.5)
        a_yz = np.random.uniform(0.5, 1.5)
        a_yx = np.random.uniform(0.3, 0.8)

        if mechanism == 'linear':
            Z = np.random.uniform(-2, 2, self.n)
            X_latent = a_xz * Z + self._noise()
            X = self._discretize(X_latent, bins=n_bins)
            if independent:
                Y = a_yz * Z + self._noise()
            else:
                Y = a_yz * Z + a_yx * X + self._noise()

        elif mechanism == 'nonlinear':
            Z = np.random.uniform(-2, 2, self.n)
            X_latent = self._random_nonlinear(a_xz * Z) + self._noise()
            X = self._discretize(X_latent, bins=n_bins)
            if independent:
                Y = self._random_nonlinear(a_yz * Z) + self._noise()
            else:
                Y = self._random_nonlinear(a_yz * Z) + a_yx * (X % 2) + self._noise()

        # Mediator (X -> Z -> Y, X discrete)
        elif mechanism == 'mediator':
            X = np.random.randint(0, n_bins, self.n)
            # X influences Z
            Z = a_xz * X + self._noise()
            if independent:
                Y = a_yz * Z + self._noise()
            else:
                Y = a_yz * Z + a_yx * X + self._noise()

        # Collider (X and Y independent, Z continuous collider)
        elif mechanism == 'collider':
            if independent:
                X = np.random.randint(0, n_bins, self.n)
                Y = np.random.normal(0, 1, self.n)
                Z = np.random.normal(0, 1, self.n)
            else:
                X = np.random.randint(0, n_bins, self.n)
                Y = np.random.normal(0, 1, self.n)
                Z = a_xz * X + a_yz * Y + self._noise()

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        return X, Y, Z

    # ------------------------------------------------------------------
    # CASE 4: DDD (all discrete)
    # ------------------------------------------------------------------
    def generate_ddd(self, mechanism='linear', independent=True):
        n_z = np.random.choice([3, 4, 5])
        n_x = np.random.choice([3, 4])
        n_y = np.random.choice([3, 4, 5])

        if mechanism == 'linear':
            Z = np.random.randint(0, n_z, size=self.n)
            noise_X = np.random.randint(0, 3, size=self.n)
            noise_Y = np.random.randint(0, 2, size=self.n)
            X = (Z + noise_X) % n_x
            if independent:
                Y = (Z + noise_Y) % n_y
            else:
                Y = (Z + X + noise_Y) % n_y

        elif mechanism == 'nonlinear':
            Z = np.random.randint(0, n_z, size=self.n)
            noise_X = np.random.randint(0, 3, size=self.n)
            noise_Y = np.random.randint(0, 2, size=self.n)
            X = (Z ** 2 + noise_X) % n_x
            if independent:
                Y = (Z ** 2 + noise_Y) % n_y
            else:
                Y = (Z * X + noise_Y) % n_y

        # Mediator: X -> Z -> Y (all discrete)
        elif mechanism == 'mediator':
            X = np.random.randint(0, n_x, self.n)
            # Z depends on X
            Z = (X + np.random.randint(0, 2, self.n)) % n_z
            if independent:
                Y = (Z + np.random.randint(0, 2, self.n)) % n_y
            else:
                Y = (Z + X + np.random.randint(0, 2, self.n)) % n_y

        # Collider: X and Y independent, Z depends on both
        elif mechanism == 'collider':
            if independent:
                X = np.random.randint(0, n_x, self.n)
                Y = np.random.randint(0, n_y, self.n)
                Z = np.random.randint(0, n_z, self.n)
            else:
                X = np.random.randint(0, n_x, self.n)
                Y = np.random.randint(0, n_y, self.n)
                Z = (X + Y + np.random.randint(0, 2, self.n)) % n_z

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        return X, Y, Z
