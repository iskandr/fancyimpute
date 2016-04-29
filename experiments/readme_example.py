import numpy as np
from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    SimpleFill
)

n = 200
m = 20
inner_rank = 4
X = np.dot(np.random.randn(n, inner_rank), np.random.randn(inner_rank, m))
print("Mean squared element: %0.4f" % (X ** 2).mean())

# X is a data matrix which we're going to randomly drop entries from
missing_mask = np.random.rand(*X.shape) < 0.1
X_incomplete = X.copy()
# missing entries indicated with NaN
X_incomplete[missing_mask] = np.nan

meanFill = SimpleFill("mean")
X_filled_mean = meanFill.complete(X_incomplete)

# Use 3 nearest rows which have a feature to fill in each row's missing features
knnImpute = KNN(k=3)
X_filled_knn = knnImpute.complete(X_incomplete)

# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
X_filled_nnm = NuclearNormMinimization().complete(X_incomplete)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
softImpute = SoftImpute()

# simultaneously normalizes the rows and columns of your observed data,
# sometimes useful for low-rank imputation methods
biscaler = BiScaler()

# rescale both rows and columns to have zero mean and unit variance
X_incomplete_normalized = biscaler.fit_transform(X_incomplete)

X_filled_softimpute_normalized = softImpute.complete(X_incomplete_normalized)
X_filled_softimpute = biscaler.inverse_transform(X_filled_softimpute_normalized)

X_filled_softimpute_no_biscale = softImpute.complete(X_incomplete)

meanfill_mse = ((X_filled_mean[missing_mask] - X[missing_mask]) ** 2).mean()
print("meanFill MSE: %f" % meanfill_mse)

# print mean squared error for the three imputation methods above
nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()
print("Nuclear norm minimization MSE: %f" % nnm_mse)

softImpute_mse = ((X_filled_softimpute[missing_mask] - X[missing_mask]) ** 2).mean()
print("SoftImpute MSE: %f" % softImpute_mse)

softImpute_no_biscale_mse = (
    (X_filled_softimpute_no_biscale[missing_mask] - X[missing_mask]) ** 2).mean()
print("SoftImpute without BiScale MSE: %f" % softImpute_no_biscale_mse)


knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
print("knnImpute MSE: %f" % knn_mse)
