# ============================================================================
# CELL 1: Imports and dependencies
# ============================================================================

import os
import sys
import time
import warnings
import math

import numpy as np
import pandas as pd

from scipy import sparse, stats
from scipy.stats import norm
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import digamma

from sklearn.neighbors import NearestNeighbors

from joblib import Parallel, delayed

# Numba is optional. If missing, fall back to no-op decorators.
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator
    prange = range

warnings.filterwarnings("ignore")
