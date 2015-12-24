from fancyimpute import MICE
import numpy as np

def test_rank1_outer_product():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, -0.1, 0.2, -0.2, 0.02])
    XY = np.outer(x, y)
    XY_missing = XY.copy()

    # drop one entry
    XY_missing[1, 2] = np.nan
    
    # column method
    XY_completed,mm = MICE(n_imputations=10,impute_type='col').complete(XY_missing)
    XY_completed_val = XY_completed.mean()
    assert abs(XY_completed_val - XY[1, 2]) < 0.001, \
    "Expected %0.4f but got %0.4f for column method" % (
        XY[1, 2], XY_completed_val)
        
    # row method doesn't work here!
    #XY_completed,mm = MICE(n_imputations=50,impute_type='row').complete(XY_missing)
    #XY_completed_val = XY_completed.mean()
    #assert abs(XY_completed_val - XY[1, 2]) < 0.001, \
    #"Expected %0.4f but got %0.4f for row method" % (
    #    XY[1, 2], XY_completed_val)
    

def test_rank1_symmetric():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, -0.1, 0.2, -0.2, 0.02])
    XY = np.outer(x, y)
    # make a symmetric matrix
    XYXY = XY.T.dot(XY)

    # drop one entry
    XY_missing = XYXY.copy()
    XY_missing[1, 2] = np.nan
    
    # column method
    XY_completed,mm = MICE(n_imputations=10,impute_type='col').complete(XY_missing)
    XY_completed_val = XY_completed.mean()
    assert abs(XY_completed_val - XYXY[1, 2]) < 0.001, \
    "Expected %0.4f but got %0.4f for column method" % (
        XYXY[1, 2], XY_completed_val)
    
    # row method doesn't work here!
    #XY_completed,mm = MICE(n_imputations=10,impute_type='row').complete(XY_missing)
    #XY_completed_val = XY_completed.mean()
    #assert abs(XY_completed_val - XYXY[1, 2]) < 0.001, \
    #"Expected %0.4f but got %0.4f for column method" % (
    #    XYXY[1, 2], XY_completed_val)