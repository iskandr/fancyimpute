# fancyimpute

Simple implementation of [Exact Matrix Completion via Convex Optimization](statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf
) by Emmanuel Candes and Benjamin Recht using cvxpy.

## Usage

```python

from fancyimpute import ConvexSolver

solver = ConvexSolver(
    min_value=0.0,
    max_value=1.0,
    error_tolerance=0.0005)

# X_incomplete has missing data which is represented with NaN values
X_filled = solver.complete(X_incomplete)
```

