

from  fancyimpute import MissForest

from low_rank_data import XY, XY_incomplete, missing_mask
from low_rank_data import XY_categorical, XY_incomplete_categorical, missing_mask_categorical
from low_rank_data import XY_mixed, XY_incomplete_mixed, missing_mask_mixed

from common import reconstruction_error

def test_missforest_all_continous():
    solver = MissForest(n_estimators=500)
    #print(XY_incomplete)
    XY_completed = solver.fit_transform(XY_incomplete) #dont need to normalization
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MissForest")
    assert missing_mae < 0.3, "Error too high!"

def test_missforest_all_categories():
    solver = MissForest(n_estimators=500)
    print(XY_incomplete_categorical)
    XY_completed = solver.fit_transform(XY_incomplete_categorical) #dont need to normalization
    print(XY_completed)
    _, missing_mae = reconstruction_error(
        XY_categorical,
        XY_completed,
        missing_mask_categorical,
        name="MissForest")

def test_missforest_mixed_type():
    solver = MissForest(n_estimators=500)
    print(XY_incomplete_mixed)
    XY_completed = solver.fit_transform(XY_incomplete_mixed) #dont need to normalization
    print(XY_completed)
    _, missing_mae = reconstruction_error(
        XY_mixed,
        XY_completed,
        missing_mask_categorical,
        name="MissForest")
