from fancyimpute import MICE
import numpy as np

from low_rank_data import create_rank_k_dataset


def test_MICE_column_with_low_rank_random_matrix():
    XY, XY_incomplete, missing_mask = create_rank_k_dataset(
        n_rows=1000,
        n_cols=20,
        k=3,
        fraction_missing=0.5)
    mice = MICE(n_imputations=100,impute_type='col')
    XY_completed_storage,mm = mice.complete(XY_incomplete,verbose=False)
    XY_completed = XY_incomplete.copy()
    XY_completed[mm] = XY_completed_storage.mean(0)
    diff = XY - XY_completed
    missing_mse = np.mean(diff[missing_mask] ** 2)
    missing_mae = np.mean(np.abs(diff[missing_mask]))
    print("MSE (col): %0.4f, MAE: %0.4f" % (missing_mse, missing_mae))
    assert missing_mae < 0.1, "Error too high with column method!"
    
def test_MICE_row_with_low_rank_random_matrix():
    XY, XY_incomplete, missing_mask = create_rank_k_dataset(
        n_rows=1000,
        n_cols=20,
        k=3,
        fraction_missing=0.5)
    mice = MICE(n_imputations=100,impute_type='row')
    XY_completed_storage,mm = mice.complete(XY_incomplete,verbose=False)
    XY_completed = XY_incomplete.copy()
    XY_completed[mm] = XY_completed_storage.mean(0)
    diff = XY - XY_completed
    missing_mse = np.mean(diff[missing_mask] ** 2)
    missing_mae = np.mean(np.abs(diff[missing_mask]))
    print("MSE (row): %0.4f, MAE: %0.4f" % (missing_mse, missing_mae))
    assert missing_mae < 0.1, "Error too high with row method!"


if __name__ == "__main__":
    test_MICE_column_with_low_rank_random_matrix()
    test_MICE_row_with_low_rank_random_matrix()