import numpy as np
from nose.tools import eq_

from fancyimpute.knn import KNN

from low_rank_data import XY, XY_incomplete, missing_mask


def test_knn():
    # get a baseline error from just zero-filling the missing entries
    sad_zero_fill = np.sum(np.abs(XY[missing_mask]))
    mad_zero_fill = sad_zero_fill / missing_mask.sum()
    print("MAD zero-fill = ", mad_zero_fill)
    for k in [5, 15, 30]:
        print("-- k=", k)
        XY_completed = KNN(k).complete(XY_incomplete)
        mask = np.isfinite(XY_completed)
        eq_((~mask).sum(), 0)
        diff = (XY_completed - XY)[missing_mask]
        sad = np.sum(np.abs(diff))
        print("Sum absolute differences", sad)
        mad = sad / missing_mask.sum()
        print("Mean absolute difference", mad)
        # knnImpute should be at least twice as good as just zero fill
        assert mad <= (mad_zero_fill / 2.0), \
            "Expected knnImpute to be 2x better than zeroFill (%f) but got MAD=%f" % (
                mad_zero_fill,
                mad)
