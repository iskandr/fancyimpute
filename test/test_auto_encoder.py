import numpy as np
from fancyimpute import AutoEncoder

from low_rank_data import create_rank_k_dataset


def test_auto_encoder_with_low_rank_random_matrix():
    XY, XY_incomplete, missing_mask = create_rank_k_dataset(
        n_rows=1000,
        n_cols=20,
        k=3,
        fraction_missing=0.5)
    XY_completed = AutoEncoder().complete(XY_incomplete)
    print(XY)
    print(XY_completed)
    print(missing_mask)
    diff = XY - XY_completed
    missing_mse = np.mean(diff[missing_mask] ** 2)
    missing_mae = np.mean(np.abs(diff[missing_mask]))
    print("MSE: %0.4f, MAE: %0.4f" % (missing_mse, missing_mae))
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_auto_encoder_with_low_rank_random_matrix()
