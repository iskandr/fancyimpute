from fancyimpute import SimpleFill

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error
from random_initialization import initialize_random_seed


def test_matrix_factorization_with_low_rank_random_matrix():
    initialize_random_seed()  # for reproducibility
    solver = SimpleFill(
        fill_method='random')
    initialize_random_seed()
    XY_completed = solver.fit(XY_incomplete, missing_mask)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")
    initialize_random_seed()
    XY_completed = solver.transform(XY_incomplete, missing_mask)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")

if __name__ == "__main__":
    test_matrix_factorization_with_low_rank_random_matrix()
