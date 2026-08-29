from mixci._compat import *  # noqa: F401,F403

from mixci.baselines import METHODS, baselines_for_regime  # noqa: F401
from mixci.generators import DataGenerator  # noqa: F401

# %% CELL 12: Trial driver and benchmark harness
# =============================================================================
def _gen_data(gen, regime, mechanism, independent):
    if regime == 'CCC':
        return gen.generate_ccc(mechanism=mechanism, independent=independent)
    if regime == 'DCC':
        return gen.generate_dc_c(mechanism=mechanism, independent=independent)
    if regime == 'CCD':
        return gen.generate_cc_d(mechanism=mechanism, independent=independent)
    if regime == 'DDD':
        return gen.generate_ddd(mechanism=mechanism, independent=independent)
    raise ValueError(f"Unknown regime: {regime}")


def run_single_trial(method_key, regime, mechanism, n, alpha, seed):
    #from data_generating import DataGenerator
    import traceback
    runner = METHODS[method_key]

    out = {'type1': np.nan, 'power': np.nan, 'runtime': np.nan,
           'error_h0': '', 'error_h1': ''}

    try:
        gen0 = DataGenerator(n_samples=n, seed=seed)
        X0, Y0, Z0 = _gen_data(gen0, regime, mechanism, True)
        t0 = time.time()
        p0 = runner(np.asarray(X0), np.asarray(Y0), np.asarray(Z0))
        out['runtime'] = time.time() - t0
        out['type1'] = int(p0 < alpha)
    except Exception as e:
        # Capture short error string: type name + first line of message.
        msg = str(e).splitlines()[0] if str(e) else ''
        out['error_h0'] = f"{type(e).__name__}: {msg}"[:200]

    try:
        gen1 = DataGenerator(n_samples=n, seed=seed + 100000)
        X1, Y1, Z1 = _gen_data(gen1, regime, mechanism, False)
        p1 = runner(np.asarray(X1), np.asarray(Y1), np.asarray(Z1))
        out['power'] = int(p1 < alpha)
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else ''
        out['error_h1'] = f"{type(e).__name__}: {msg}"[:200]

    return out


# ============================================================================
# Benchmark driver
# ============================================================================

def run_benchmark(
    sample_sizes=(500, 1000, 2000),
    mechanisms=('linear', 'nonlinear'),
    regimes=('CCC', 'DCC', 'CCD', 'DDD'),
    n_trials=100,
    alpha=0.05,
    n_jobs=-1,
    output_dir='results',
    slow_methods=('ours_debiased_perm',),
    slow_n_trials=30,
    verbose=True,
):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    raw_rows = []

    for n in sample_sizes:
        if verbose:
            print(f"\n{'=' * 65}\nN = {n}\n{'=' * 65}")
        for regime in regimes:
            method_list = baselines_for_regime(regime)
            for mech in mechanisms:
                for method in method_list:
                    n_t = slow_n_trials if method in slow_methods else n_trials
                    if verbose:
                        print(f"  {regime:>4s} {mech:>10s} {method:>22s} "
                              f"N={n} (trials={n_t})...",
                              end='', flush=True)
                    t_start = time.time()
                    results = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(run_single_trial)(
                            method, regime, mech, n, alpha, seed
                        )
                        for seed in range(n_t)
                    )
                    elapsed = time.time() - t_start

                    type1_arr = np.array([r['type1'] for r in results],
                                          dtype=float)
                    power_arr = np.array([r['power'] for r in results],
                                          dtype=float)
                    rt_arr = np.array([r['runtime'] for r in results],
                                       dtype=float)

                    t1 = float(np.nanmean(type1_arr))
                    pw = float(np.nanmean(power_arr))
                    rt = float(np.nanmean(rt_arr))

                    # Count failures and capture the first non-empty
                    # error message so failures are visible.
                    n_fail_h0 = sum(1 for r in results if r['error_h0'])
                    n_fail_h1 = sum(1 for r in results if r['error_h1'])
                    first_err = next((r['error_h0'] for r in results
                                      if r['error_h0']), '') or \
                                next((r['error_h1'] for r in results
                                      if r['error_h1']), '')

                    if verbose:
                        print(f" T1={t1:.3f} Pow={pw:.3f} RT={rt:.3f}s "
                              f"(wall={elapsed:.1f}s)")
                        if first_err and (n_fail_h0 + n_fail_h1) > 0:
                            print(f"      [!] failures: H0={n_fail_h0}/{n_t}, "
                                  f"H1={n_fail_h1}/{n_t}; first error: "
                                  f"{first_err}")

                    rows.append({
                        'N': n, 'Regime': regime, 'Mechanism': mech,
                        'Method': method,
                        'Type1': t1, 'Power': pw,
                        'AvgRuntime': rt,
                        'Trials': n_t,
                    })

                    for ti, r in enumerate(results):
                        raw_rows.append({
                            'N': n, 'Regime': regime, 'Mechanism': mech,
                            'Method': method, 'Trial': ti,
                            'Type1': r['type1'], 'Power': r['power'],
                            'Runtime': r['runtime'],
                            'ErrorH0': r['error_h0'],
                            'ErrorH1': r['error_h1'],
                        })

        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, 'benchmark_debiased_summary.csv'),
            index=False
        )
        pd.DataFrame(raw_rows).to_csv(
            os.path.join(output_dir, 'benchmark_debiased_raw.csv'),
            index=False
        )

    df = pd.DataFrame(rows)
    runtime_pivot = df.pivot_table(
        index=['Regime', 'Mechanism', 'Method'], columns='N',
        values='AvgRuntime', aggfunc='mean'
    ).round(4)
    runtime_pivot.to_csv(
        os.path.join(output_dir, 'runtime_debiased_summary.csv')
    )

    # Summarize failures by method
    raw_df = pd.DataFrame(raw_rows)
    if len(raw_df) > 0:
        raw_df['failed_h0'] = raw_df['ErrorH0'].astype(bool)
        raw_df['failed_h1'] = raw_df['ErrorH1'].astype(bool)
        fail_summary = raw_df.groupby('Method')[
            ['failed_h0', 'failed_h1']
        ].mean()
        problem_methods = fail_summary[
            (fail_summary['failed_h0'] > 0.10)
            | (fail_summary['failed_h1'] > 0.10)
        ]
        if verbose and len(problem_methods) > 0:
            print("\n[!] Methods with >10% trial failures:")
            print(problem_methods.round(3).to_string())
            print(
                "\n    Inspect the 'ErrorH0' / 'ErrorH1' columns of "
                "benchmark_debiased_raw.csv to diagnose."
            )
        elif verbose:
            print("\n[OK] No method had >10% failures across the grid.")

    if verbose:
        print(f"\n[Done] Results in '{output_dir}/'")
    return df
