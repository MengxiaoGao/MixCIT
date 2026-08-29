"""
mixci
=====

Debiased conditional-independence testing (MixCI) for mixed
continuous/discrete data, with a Regime-IV specialization
(X continuous, Z discrete), baselines, synthetic data generators,
and a benchmarking harness.
"""

from mixci.core import MixCIDebiased  # noqa: F401

# Populates MixCIDebiased with its methods (neighborhood construction,
# the debiased test, diagnostics, run_ci_test, ...). Must run before
# MixCIDebiased is used.
from mixci import _patch  # noqa: F401

from mixci.generators import DataGenerator  # noqa: F401
from mixci.baselines import METHODS, baselines_for_regime  # noqa: F401
from mixci.benchmark import run_benchmark, run_single_trial  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "MixCIDebiased",
    "DataGenerator",
    "METHODS",
    "baselines_for_regime",
    "run_benchmark",
    "run_single_trial",
]
