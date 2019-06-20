[![Build Status](https://travis-ci.org/iskandr/fancyimpute.svg?branch=master)](https://travis-ci.org/iskandr/fancyimpute) [![Coverage Status](https://coveralls.io/repos/github/iskandr/fancyimpute/badge.svg?branch=master)](https://coveralls.io/github/iskandr/fancyimpute?branch=master) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.51773.svg)](http://dx.doi.org/10.5281/zenodo.51773)


# fancyimpute

A variety of matrix completion and imputation algorithms implemented in Python 3.6.

## Usage

```python
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

# X is the complete data matrix
# X_incomplete has the same values as X except a subset have been replace with NaN

# Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=3).fit_transform(X_incomplete)

# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
X_filled_nnm = NuclearNormMinimization().fit_transform(X_incomplete)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
X_incomplete_normalized = BiScaler().fit_transform(X_incomplete)
X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)

# print mean squared error for the  imputation methods above
nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()
print("Nuclear norm minimization MSE: %f" % nnm_mse)

softImpute_mse = ((X_filled_softimpute[missing_mask] - X[missing_mask]) ** 2).mean()
print("SoftImpute MSE: %f" % softImpute_mse)

knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
print("knnImpute MSE: %f" % knn_mse)
```

## Algorithms

* `SimpleFill`: Replaces missing entries with the mean or median of each column.

* `KNN`: Nearest neighbor imputations which weights samples using the mean squared difference
on features for which two rows both have observed data.

* `SoftImpute`: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the [softImpute](https://web.stanford.edu/~hastie/swData/softImpute/vignette.html) package for R, which is based on [Spectral Regularization Algorithms for Learning Large Incomplete Matrices](http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf) by Mazumder et. al.

* `IterativeSVD`: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from [Missing value estimation methods for DNA microarrays](http://www.ncbi.nlm.nih.gov/pubmed/11395428) by Troyanskaya et. al.

* `MatrixFactorization`: Direct factorization of the incomplete matrix into low-rank `U` and `V`, with an L1 sparsity penalty on the elements of `U` and an L2 penalty on the elements of `V`. Solved by gradient descent.

* `NuclearNormMinimization`: Simple implementation of [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf
) by Emmanuel Candes and Benjamin Recht using [cvxpy](http://www.cvxpy.org). Too slow for large matrices.

* `BiScaler`: Iterative estimation of row/column means and standard deviations to get doubly normalized
matrix. Not guaranteed to converge but works well in practice. Taken from [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](http://arxiv.org/abs/1410.2596).
