import climate
import downhill
import numpy as np
import theano
import theano.tensor as T


class MatrixFactorization(object):
    """
    Given an incomplete (m,n) matrix X, factorize it into
    U, V where U.shape = (m, k) and V.shape = (k, n).

    The U, V are found by minimizing the difference between U.dot.V and
    X at the observed entries along with a sparsity penalty for U and an
    L2 penalty for V.
    """
    def __init__(
            self,
            k=10,
            initializer=np.random.randn,
            learning_rate=0.01,
            patience=5,
            l1_penalty_weight=0.1,
            l2_penalty_weight=0.1,
            max_gradient_norm=5):
        self.k = k
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.patience = patience
        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight
        self.max_gradient_norm = max_gradient_norm
        climate.enable_default_logging()

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def complete(self, X, verbose=True):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        self._check_input(X)
        (n_samples, n_features) = X.shape

        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)

        X = X.copy()
        # replace NaN's with 0
        X[missing_mask] = 0

        observed_mask = 1 - missing_mask

        # Set up a matrix factorization problem to optimize.
        U_init = self.initializer(n_samples, self.k).astype(X.dtype)
        V_init = self.initializer(self.k, n_features).astype(X.dtype)
        U = theano.shared(U_init, name="U")
        V = theano.shared(V_init, name="V")
        X_symbolic = T.matrix(name="X", dtype=X.dtype)
        reconstruction = T.dot(U, V)

        difference = X_symbolic - reconstruction

        masked_difference = difference * observed_mask
        err = T.sqr(masked_difference)
        mse = err.mean()
        loss = (
            mse +
            self.l1_penalty_weight * abs(U).mean() +
            self.l2_penalty_weight * (V * V).mean()
        )
        downhill.minimize(
            loss=loss,
            train=[X],
            patience=self.patience,
            batch_size=n_samples,
            max_gradient_norm=self.max_gradient_norm,  # Prevent gradient explosion!
            learning_rate=self.learning_rate,
            monitors=(('err', err.mean()),    # Monitor during optimization.
                      ('|u|<0.1', (abs(U) < 0.1).mean()),
                      ('|v|<0.1', (abs(U) < 0.1).mean())),
            monitor_gradients=True)

        U_value = U.get_value()
        V_value = V.get_value()
        X_full = np.dot(U_value, V_value)
        return X_full
