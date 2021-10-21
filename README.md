[![Build Status](https://travis-ci.org/iskandr/fancyimpute.svg?branch=master)](https://travis-ci.org/iskandr/fancyimpute) [![Coverage Status](https://coveralls.io/repos/github/iskandr/fancyimpute/badge.svg?branch=master)](https://coveralls.io/github/iskandr/fancyimpute?branch=master) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.51773.svg)](http://dx.doi.org/10.5281/zenodo.51773)


![plot](https://user-images.strikinglycdn.com/res/hrscywv4p/image/upload/c_limit,fl_lossy,h_1440,w_720,f_auto,q_auto/174108/654579_364680.png)


A variety of matrix completion and imputation algorithms implemented in Python 3.6.

To install:

`pip install fancyimpute`

If you run into `tensorflow` problems and use anaconda, you can try to fix them with `conda install cudatoolkit`. 

## Important Caveats

(1) This project is in "bare maintenance" mode. That means we are not planning on adding more imputation algorithms or features (but might if we get inspired). Please do report bugs, and we'll try to fix them. Also, we are happy to take pull requests for more algorithms and/or features. 

(2) `IterativeImputer` started its life as a `fancyimpute` original, but was then merged into `scikit-learn` and we deleted it from `fancyimpute` in favor of the better-tested `sklearn` version. As a convenience, you can still `from fancyimpute import IterativeImputer`, but under the hood it's just doing `from sklearn.impute import IterativeImputer`.  That means if you update `scikit-learn` in the future, you may also change the behavior of `IterativeImputer`. 


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

* `IterativeImputer`: A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion. A stub that links to `scikit-learn`'s [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html).

* `IterativeSVD`: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from [Missing value estimation methods for DNA microarrays](http://www.ncbi.nlm.nih.gov/pubmed/11395428) by Troyanskaya et. al.

* `MatrixFactorization`: Direct factorization of the incomplete matrix into low-rank `U` and `V`, with per-row and per-column biases, as well as a global bias. Solved by SGD in pure numpy.

* `NuclearNormMinimization`: Simple implementation of [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf
) by Emmanuel Candes and Benjamin Recht using [cvxpy](http://www.cvxpy.org). Too slow for large matrices.

* `BiScaler`: Iterative estimation of row/column means and standard deviations to get doubly normalized
matrix. Not guaranteed to converge but works well in practice. Taken from [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](http://arxiv.org/abs/1410.2596).

## Citation

If you use `fancyimpute` in your academic publication, please cite it as follows:
```bibtex
@software{fancyimpute,
  author = {Alex Rubinsteyn and Sergey Feldman},
  title={fancyimpute: An Imputation Library for Python},
  url = {https://github.com/iskandr/fancyimpute},
  version = {0.7.0},
  date = {2016},
}
```
