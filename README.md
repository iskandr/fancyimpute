# fancyimpute

A variety of matrix completion and imputation algorithms implemented in Python.

## Usage

```python

from fancyimpute import NuclearNormMinimization

solver = NuclearNormMinimization(
    min_value=0.0,
    max_value=1.0,
    error_tolerance=0.0005)

# X_incomplete has missing data which is represented with NaN values
X_filled = solver.complete(X_incomplete)
```
## Algorithms

* `SimpleFill`: Replaces missing entries with the mean or median of each column.

* `SoftImpute`: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the [softImpute](https://web.stanford.edu/~hastie/swData/softImpute/vignette.html) package for R, which is based on [Spectral Regularization Algorithms for Learning Large Incomplete Matrices](http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf) by Mazumder et. al.

* `IterativeSVD`: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from [Missing value estimation methods for DNA microarrays](http://www.ncbi.nlm.nih.gov/pubmed/11395428) by Troyanskaya et. al.

* `MICE`: Reimplementation of [Multiple Imputation by Chained Equations](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/).

* `MatrixFactorization`: Direct factorization of the incomplete matrix into low-rank `U` and `V`, with an L1 sparsity penalty on the elements of `U` and an L2 penalty on the elements of `V`. Solved by gradient descent.

* `NuclearNormMinimization`: Simple implementation of [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf
) by Emmanuel Candes and Benjamin Recht using [cvxpy](http://www.cvxpy.org/en/latest/). Too slow for large matrices.

