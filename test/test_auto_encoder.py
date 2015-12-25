from fancyimpute import AutoEncoder
import numpy as np

x = np.arange(1000) + 1
y = np.sqrt(10 + np.arange(1000))

XY = np.outer(x, y)
XY_missing = XY.copy()
# drop one entry
XY_missing[1, 2] = np.nan


def test_rank1_auto_encoder():
    XY_completed = AutoEncoder().complete(XY_missing)
    assert abs(XY_completed[1, 2] - XY[1, 2]) < 0.001, \
        "Expected %0.4f but got %0.4f" % (
            XY[1, 2], XY_completed[1, 2])
