from fancyimpute import AutoEncoder

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def test_auto_encoder_with_low_rank_random_matrix():
    solver = AutoEncoder(
        hidden_layer_sizes=None,
        hidden_activation="tanh",
        optimizer="adam",
        recurrent_weight=0.0)
    XY_completed = solver.complete(
        XY_incomplete)
    _, missing_mae = reconstruction_error(XY, XY_completed, missing_mask)
    assert missing_mae < 0.1, "Error too high!"

"""
def test_auto_encoder_with_low_rank_random_matrix_using_hallucination():
    solver = AutoEncoder(
        hidden_layer_sizes=[3],
        hidden_activation="tanh",
        optimizer="adam",
        recurrent_weight=0.5)
    XY_completed = solver.complete(
        XY_incomplete)
    _, missing_mae = reconstruction_error(XY, XY_completed, missing_mask)
    assert missing_mae < 0.1, "Error too high!"
"""
if __name__ == "__main__":
    test_auto_encoder_with_low_rank_random_matrix()
    """
    test_auto_encoder_with_low_rank_random_matrix_using_hallucination()
    """
