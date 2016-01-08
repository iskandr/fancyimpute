from fancyimpute import MatrixFactorization

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def test_matrix_factorization_with_low_rank_random_matrix():
    solver = MatrixFactorization(
        rank=3,
        l1_penalty=0,
        l2_penalty=0)
    XY_completed = solver.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")
    assert missing_mae < 0.01, "Error too high!"

if __name__ == "__main__":
    test_matrix_factorization_with_low_rank_random_matrix()
