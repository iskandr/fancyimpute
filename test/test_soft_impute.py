from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error

from fancyimpute import SoftImpute

def test_soft_impute_with_low_rank_random_matrix():
    solver = SoftImpute()
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="SoftImpute")
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_soft_impute_with_low_rank_random_matrix()
