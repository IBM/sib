# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from sib.c_package.c_sib_optimizer_sparse import \
    CSIBOptimizerSparseInt as CSIBOptimizerInt, \
    CSIBOptimizerSparseFloat as CSIBOptimizerFloat
import numpy as np


class CSIBOptimizerSparse:
    def __init__(self, n_clusters, n_features, n_samples, xy, xy_sum, x_sum):
        if np.issubdtype(xy_sum.dtype, np.integer):
            self.c_sib_optimizer = CSIBOptimizerInt(n_clusters, n_features)
        else:
            self.c_sib_optimizer = CSIBOptimizerFloat(n_clusters, n_features)
        self.n_samples = n_samples
        self.xy = xy
        self.xy_sum = xy_sum
        self.x_sum = x_sum

    def optimize(self, x_permutation, t_size, t_sum, t_log_sum, t_centroid, labels, ity):
        return self.c_sib_optimizer.optimize(self.n_samples, self.xy.indices,
                                             self.xy.indptr, self.xy.data, self.xy_sum,
                                             self.x_sum, x_permutation, t_size, t_sum,
                                             t_log_sum, t_centroid, labels, ity)

    def infer(self, n_samples, xy, xy_sum, x_sum, t_size, t_sum, t_log_sum, t_centroid, labels, costs):
        return self.c_sib_optimizer.infer(n_samples, xy.indices, xy.indptr, xy.data, xy_sum, x_sum,
                                          t_size, t_sum, t_log_sum, t_centroid, labels, costs)
