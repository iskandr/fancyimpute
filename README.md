# fancyimpute

A variety of matrix completion and imputation algorithms implemented in Python.

## Usage

```python
from fancyimpute import (NuclearNormMinimization, BiScaler, DenseKNN)

biscaler = BiScaler()

# X is a data matrix which we're going to randomly drop entries from
missing_mask = np.random.randn(*X.shape) > 0
X_incomplete = X.copy()
# missing entries indicated with NaN
X_incomplete[missing_mask] = np.nan

# rescale both rows and columns to have zero mean and unit variance
X_incomplete_normalized = biscaler.fit_transform(X_incomplete)

# use 3 nearest rows which have a feature to fill in each row's missing features
knn_solver = DenseKNN(k=3)
X_filled_normalized = knn_solver.complete(X_incomplete)
X_filled = biscaler.inverse_transform(X_knn_normalized)

mse = ((X_filled[missing_mask] - X[missing_mask]) ** 2).mean()
print("MSE of reconstruction: %f" % mse)
```

## Algorithms

* `SimpleFill`: Replaces missing entries with the mean or median of each column.

* `DenseKNN`: Nearest neighbor imputations which weights samples using the mean squared difference
on features for which two rows both have observed data.

* `SoftImpute`: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the [softImpute](https://web.stanford.edu/~hastie/swData/softImpute/vignette.html) package for R, which is based on [Spectral Regularization Algorithms for Learning Large Incomplete Matrices](http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf) by Mazumder et. al.

* `IterativeSVD`: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from [Missing value estimation methods for DNA microarrays](http://www.ncbi.nlm.nih.gov/pubmed/11395428) by Troyanskaya et. al.

* `MICE`: Reimplementation of [Multiple Imputation by Chained Equations](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/).

* `MatrixFactorization`: Direct factorization of the incomplete matrix into low-rank `U` and `V`, with an L1 sparsity penalty on the elements of `U` and an L2 penalty on the elements of `V`. Solved by gradient descent.

* `NuclearNormMinimization`: Simple implementation of [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf
) by Emmanuel Candes and Benjamin Recht using [cvxpy](http://www.cvxpy.org/en/latest/). Too slow for large matrices.

* `BiScaler`: Iterative estimation of row/column means and standard deviations to get doubly normalized
matrix. Not guaranteed to converge but works well in practice. Taken from [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](http://arxiv.org/abs/1410.2596).

