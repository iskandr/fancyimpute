class WeightedNearestColumns(object):
    """
    Fill in missing each missing (row, column) value by averaging across the
    k-nearest neighbors columns which are not missing that row.
    """

    def __init__(self, k=None):
        """
        Parameters
        ----------
        k : int, optional
            If omitted, then average across all columns.
        """
        self.k = k


