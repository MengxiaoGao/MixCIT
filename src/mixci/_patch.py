from mixci.core import MixCIDebiased

from mixci.neighborhood import _build_neighborhood, _gather_kernel
from mixci.debiased import _debiased_test
from mixci.marginal import _marginal_test
from mixci.diagnostics import (
    _select_polynomial_order,
    _diagnose_design_conditioning,
    _stratified_permutation,
    _permutation_calibrate,
    debiased_test_with_diagnostics,
)
from mixci.api import run_ci_test

MixCIDebiased._build_neighborhood = _build_neighborhood
MixCIDebiased._gather_kernel = _gather_kernel
MixCIDebiased._debiased_test = _debiased_test
MixCIDebiased._marginal_test = _marginal_test
MixCIDebiased._select_polynomial_order = _select_polynomial_order
MixCIDebiased._diagnose_design_conditioning = _diagnose_design_conditioning
MixCIDebiased._stratified_permutation = _stratified_permutation
MixCIDebiased._permutation_calibrate = _permutation_calibrate
MixCIDebiased.debiased_test_with_diagnostics = debiased_test_with_diagnostics
MixCIDebiased.run_ci_test = run_ci_test
