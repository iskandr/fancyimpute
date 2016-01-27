from time import time
import numpy as np
from fancyimpute.knn_helpers import (
    knn_impute_optimistic,
    knn_impute_with_argpartition,
    knn_impute_few_observed,
    knn_impute_reference,
)

if __name__ == "__main__":
    for fraction_missing in [0.95, 0.25, 0.5, 0.75]:
        for n_rows in [100, 1000]:
            for n_cols in [100, 500]:
                for k in [15]:
                    print("-- Fraction=%0.2f, n_rows=%d, n_cols=%d, k=%d" % (
                        fraction_missing,
                        n_rows,
                        n_cols,
                        k))
                    X = np.random.randn(n_rows, n_cols)
                    missing_mask = np.random.rand(n_rows, n_cols) < fraction_missing
                    X[missing_mask] = np.nan
                    start_t = time()
                    knn_impute_optimistic(X, missing_mask, k, verbose=False)
                    end_t = time()
                    print("OPTIMISTIC TIME: %0.4f" % (end_t - start_t))

                    X[missing_mask] = np.nan
                    start_t = time()
                    knn_impute_with_argpartition(X, missing_mask, k, verbose=False)
                    end_t = time()
                    print("ARGPARTITION TIME: %0.4f" % (end_t - start_t))

                    X[missing_mask] = np.nan
                    start_t = time()
                    knn_impute_few_observed(X, missing_mask, k, verbose=False)
                    end_t = time()
                    print("SPARSE TIME: %0.4f" % (end_t - start_t))

                    X[missing_mask] = np.nan
                    start_t = time()
                    knn_impute_reference(X, missing_mask, k, verbose=False)
                    end_t = time()
                    print("REFERENCE TIME: %0.4f" % (end_t - start_t))
