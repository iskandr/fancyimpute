

from .solver import Solver

class IterativeSVD(Solver):
    def __init__(self, k):
        self.k = k

    def complete(X):
        1. Let A(t) = PΩA + PΩc M(t).
        2. Set M(t+1) to be the rank-q SVD of A(t).