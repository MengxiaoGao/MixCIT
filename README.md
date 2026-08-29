# mixci

Debiased conditional-independence testing for mixed continuous/discrete
data.

## Layout

```
src/mixci/
    _compat.py      # numpy/scipy/sklearn/joblib imports + numba fallback
    monomials.py     # numba helpers: monomial design, per-anchor solver
    core.py          # MixCIDebiased class (constructor, dispatch)
    variance.py       # overlap-graph variance helpers
    neighborhood.py    # k-NN neighborhood construction, kernel gathering
    debiased.py         # debiased test incl. Regime IV specialization
    marginal.py          # marginal (empty-Z) test
    diagnostics.py        # polynomial-order selection, permutation calibration
    api.py                 # run_ci_test (full-diagnostics entry point)
    generators.py           # DataGenerator (CCC/DCC/CCD/DDD regimes)
    baselines.py             # baseline CI tests (KCI, FisherZ, CMIknn, chi2, G-test...)
    benchmark.py              # trial driver + run_benchmark harness
    cli.py                     # `mixci-benchmark` console entry point
    _patch.py                   # wires the above methods onto MixCIDebiased
```

`_patch.py` reproduces, once and deterministically, the monkey-patching
the original notebook did across cells (`MixCIDebiased._x = _x`, ...).
Behavior is unchanged from the notebook — only the file organization.

## Install

```bash
pip install -e .            # core deps only (numba-free fallback)
pip install -e ".[numba]"   # + numba acceleration (recommended)
pip install -e ".[dev]"     # + pytest
```

## Use

```python
import numpy as np
from mixci import MixCIDebiased, DataGenerator, run_benchmark

gen = DataGenerator(n_samples=1000, seed=0)
X, Y, Z = gen.generate_ccc(mechanism="linear", independent=True)

test = MixCIDebiased()
result = test.run_ci_test(0, 1, [], return_full=True)  # example call shape;
                                                          # see api.py for the
                                                          # real signature
```

Run the full benchmark used in the original notebook's `__main__` block:

```bash
mixci-benchmark
```

or from Python:

```python
from mixci import run_benchmark
df = run_benchmark(
    sample_sizes=(500, 1000, 2000),
    mechanisms=("linear", "nonlinear"),
    regimes=("CCC", "DCC", "CCD", "DDD"),
    n_trials=100,
)
```

## Notes

- Numba is optional. Without it, `_compat.py` falls back to no-op
  `njit`/`prange` shims (pure-Python, slower).
- This refactor did not change any algorithmic code — it only moved
  cell bodies into modules and centralized the monkey-patch calls in
  `_patch.py`. If you find a bug, it very likely also exists in the
  original notebook.
