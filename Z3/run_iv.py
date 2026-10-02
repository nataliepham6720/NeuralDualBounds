"""
IV_cont ATE bounds through every route, on the outcome scale.

    python -m Z3.run_iv --k 6
    python -m Z3.run_iv --k 10 --routes reduced-highs reduced-scip reduced
    python -m Z3.run_iv --k 6 --units bins     # the bin-index scale
"""

import argparse
import os

import numpy as np

from Data.IV_cont.LP_construction import (empirical_distribution_IV,
                                          generate_data_IV)

from . import problems, reference, z3_oracle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES = ("primal", "full-highs", "full-scip", "full-ipm",
          "reduced-highs", "reduced-scip", "reduced")
TRUE_ATE = 3.0


def load_iv(k, n=10000, lam=0.5, seed=2020):
    """``(P, edges, source)``: P(T, Y | Z) on ``k`` bins and the Y-bin edges."""
    np.random.seed(seed)
