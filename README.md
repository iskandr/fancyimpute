[![Build Status](https://travis-ci.org/iskandr/fancyimpute.svg?branch=master)](https://travis-ci.org/iskandr/fancyimpute) [![Coverage Status](https://coveralls.io/repos/github/iskandr/fancyimpute/badge.svg?branch=master)](https://coveralls.io/github/iskandr/fancyimpute?branch=master) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.51773.svg)](http://dx.doi.org/10.5281/zenodo.51773)


# fancyimpute

A variety of matrix completion and imputation algorithms implemented in Python.

## Usage

```python
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler

# X is the complete data matrix
# X_incomplete has the same values as X except a subset have been replace with NaN

# Model each feature with missing values as a function of other features, and
# use that estimate for imputation.
X_filled_ii = IterativeImputer().fit_transform(X_incomplete)

# Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=3).fit_transform(X_incomplete)

# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
X_filled_nnm = NuclearNormMinimization().fit_transform(X_incomplete)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
X_incomplete_normalized = BiScaler().fit_transform(X_incomplete)
X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)

# print mean squared error for the four imputation methods above
ii_mse = ((X_filled_ii[missing_mask] - X[missing_mask]) ** 2).mean()
print("Iterative Imputer norm minimization MSE: %f" % ii_mse)

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

* `IterativeImputer`: A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.

* `MatrixFactorization`: Direct factorization of the incomplete matrix into low-rank `U` and `V`, with an L1 sparsity penalty on the elements of `U` and an L2 penalty on the elements of `V`. Solved by gradient descent.

* `NuclearNormMinimization`: Simple implementation of [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf
) by Emmanuel Candes and Benjamin Recht using [cvxpy](http://www.cvxpy.org). Too slow for large matrices.

* `BiScaler`: Iterative estimation of row/column means and standard deviations to get doubly normalized
matrix. Not guaranteed to converge but works well in practice. Taken from [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](http://arxiv.org/abs/1410.2596).

## Note about Inductive vs Transductive Imputation
Most imputation algorithms in `fancyimpute` are *transductive*. In the elegant language of `scikit-learn`'s API
this means that you can only call `solver.fit_transform(X_incomplete)`, but then the "fitted" `solver` will not
be able to be applied to new data via a call to `solver.transform`. A simple example is the `MatrixFactorization`
imputer, which decomposes as follows: `<A,B> = X_incomplete`, such that the product of `A` and `B` is close
to `X_incomplete` on its non-missing values. How then, can we apply the learned `A` and `B` matrices to
held-out data? It is not doable in general, but there are special cases. `fancyimpute` aims to be of general
use and we have not implemented an inductive mode for `MatrixFactorization`.

There are some imputation algorithms that are *inductive*, meaning they can be applied to new data after a call to
`solver.fit` or `solver.fit_transform`. Currently only `IterativeImputer` supports the full `scikit-learn` API: `fit`, `fit_transform`,
and `transform`, but we are actively looking for contributions that extend other imputers to support
induction. At least the `KNN` and `SimpleFill` imputers can be extended in a straightforward manner.

## Note about Multiple vs. Single Imputation
(From `scikit-learn`'s documentation)

In the statistics community, it is common practice to perform multiple imputations,
generating, for example, ``m`` separate imputations for a single feature matrix.
Each of these ``m`` imputations is then put through the subsequent analysis pipeline
(e.g. feature engineering, clustering, regression, classification). The ``m`` final
analysis results (e.g. held-out validation errors) allow the data scientist
to obtain understanding of how analytic results may differ as a consequence
of the inherent uncertainty caused by the missing values. The above practice
is called multiple imputation.

Our implementation of `IterativeImputer` was inspired by the R MICE
package (Multivariate Imputation by Chained Equations) [1], but differs from
it by returning a single imputation instead of multiple imputations.  However,
IterativeImputer` can also be used for multiple imputations by applying
it repeatedly to the same dataset with different random seeds when
``sample_posterior=True``. A quick example:

```python
import numpy as np
from fancyimpute import IterativeImputer

XY_incomplete = ... # insert your data here

n_imputations = 5
XY_completed = []
for i in range(n_imputations):
    imputer = IterativeImputer(n_iter=5, sample_posterior=True, random_state=i)
    XY_completed.append(imputer.fit_transform(XY_incomplete))

XY_completed_mean = np.mean(XY_completed, 0)
XY_completed_std = np.std(XY_completed, 0)
```

See [2], chapter 4 for more discussion on multiple
vs. single imputations.

It is still an open problem as to how useful single vs. multiple imputation is in
the context of prediction and classification when the user is not interested in
measuring uncertainty due to missing values.

[1] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice: Multivariate
   Imputation by Chained Equations in R". Journal of Statistical Software 45:
   1-67.

[2] Roderick J A Little and Donald B Rubin (1986). "Statistical Analysis
    with Missing Data". John Wiley & Sons, Inc., New York, NY, USA.