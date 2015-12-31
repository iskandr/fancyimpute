from fancyimpute import AutoEncoder

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def test_auto_encoder_with_low_rank_random_matrix():
    XY_completed = AutoEncoder().complete(XY_incomplete)
    _, missing_mae = reconstruction_error(XY, XY_completed, missing_mask)
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_auto_encoder_with_low_rank_random_matrix()
