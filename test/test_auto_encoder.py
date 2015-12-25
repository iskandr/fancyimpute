from fancyimpute import AutoEncoder

from rank1_data import create_rank1_dataset


def test_rank1_auto_encoder():
    XY, XY_incomplete, missing_mask = create_rank1_dataset(n=5)
    XY_completed = AutoEncoder().complete(XY_incomplete)
    print(XY)
    print(missing_mask)
    print(XY_completed)
    for i in range(XY.shape[0]):
        for j in range(XY.shape[1]):
            expected = XY[i, j]
            predicted = XY_completed[i, j]
            assert abs(expected - predicted) < 0.1, \
                "Expected %0.4f but got %0.4f at XY[%d,%d]" % (
                    expected,
                    predicted,
                    i,
                    j)


if __name__ == "__main__":
    test_rank1_auto_encoder()
