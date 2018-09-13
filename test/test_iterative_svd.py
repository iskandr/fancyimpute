from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error

from fancyimpute import IterativeSVD

def test_iterative_svd_with_low_rank_random_matrix():
    solver = IterativeSVD(rank=3)
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="IterativeSVD")
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_iterative_svd_with_low_rank_random_matrix()
