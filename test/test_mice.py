from fancyimpute import MICE

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def test_mice_column_with_low_rank_random_matrix():
    mice = MICE(n_imputations=100, impute_type='col', approximate_but_fast_mode=False)
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=col)")
    assert missing_mae < 0.1, "Error too high with column method!"


def test_mice_row_with_low_rank_random_matrix():
    mice = MICE(n_imputations=100, impute_type='row', approximate_but_fast_mode=False)
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=row)")
    assert missing_mae < 0.1, "Error too high with row method!"


def test_mice_row_with_low_rank_random_matrix_approximate():
    mice = MICE(n_imputations=100, impute_type='row', approximate_but_fast_mode=True)
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=row)")
    assert missing_mae < 0.1, "Error too high with row method!"


def test_mice_column_with_low_rank_random_matrix_approximate():
    mice = MICE(n_imputations=100, impute_type='col', approximate_but_fast_mode=True)
    XY_completed = mice.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MICE (impute_type=col)")
    assert missing_mae < 0.1, "Error too high with column method!"


if __name__ == "__main__":
    test_mice_column_with_low_rank_random_matrix()
    test_mice_row_with_low_rank_random_matrix()
