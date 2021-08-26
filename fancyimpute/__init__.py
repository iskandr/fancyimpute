from __future__ import absolute_import, print_function, division

from .solver import Solver
from .nuclear_norm_minimization import NuclearNormMinimization
from .matrix_factorization import MatrixFactorization
from .iterative_svd import IterativeSVD
from .simple_fill import SimpleFill
from .soft_impute import SoftImpute
from .scaler import BiScaler
from .knn import KNN
from .similarity_weighted_averaging import SimilarityWeightedAveraging

# while iterative imputer is experimental in sklearn, we need this
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

__version__ = "0.6.1"

__all__ = [
    "Solver",
    "NuclearNormMinimization",
    "MatrixFactorization",
    "IterativeSVD",
    "SimpleFill",
    "SoftImpute",
    "BiScaler",
    "KNN",
    "SimilarityWeightedAveraging",
    "IterativeImputer",
]
