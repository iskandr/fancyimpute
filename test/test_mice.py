from fancyimpute import MICE

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def test_mice_column_with_low_rank_random_matrix():
    mice = MICE(n_imputations=100, impute_type='col')
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=col)")
    assert missing_mae < 0.1, "Error too high with column method!"


def test_mice_row_with_low_rank_random_matrix():
    mice = MICE(n_imputations=100, impute_type='pmm')
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=row)")
    assert missing_mae < 0.1, "Error too high with PMM method!"


def test_mice_column_with_low_rank_random_matrix_approximate():
    mice = MICE(n_imputations=100, impute_type='col', n_nearest_columns=5)
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=col)")
    assert missing_mae < 0.1, "Error too high with approximate column method!"


def test_mice_row_with_low_rank_random_matrix_approximate():
    mice = MICE(n_imputations=100, impute_type='pmm', n_nearest_columns=5)
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=row)")
    assert missing_mae < 0.1, "Error too high with approximate PMM method!"


if __name__ == "__main__":
    test_mice_column_with_low_rank_random_matrix()
    test_mice_row_with_low_rank_random_matrix()
    test_mice_column_with_low_rank_random_matrix_approximate()
    test_mice_row_with_low_rank_random_matrix_approximate()
