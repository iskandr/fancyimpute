TODO: Make iterative rank increase numerically stable when ```int``` type is overloaded by ```np.int32```. (Stability improvement fro large ranks)

TODO: Do not use wrapped ```TruncatedSVD```, but use ```randomized_svd``` from scipy package directly. (Speed improvement; May cause stability issues)

TODO: Keep in memory data from SVD and elaborate ```fit``` function.
