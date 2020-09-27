# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.sparse import issparse

from sib.c_package.c_sib_optimizer import \
    CSIBOptimizerInt as CSIBOptimizerInt, \
    CSIBOptimizerFloat as CSIBOptimizerFloat


class CSIBOptimizer:
    def __init__(self, n_clusters, n_features, n_samples, xy, xy_sum, x_sum):
        if np.issubdtype(xy_sum.dtype, np.integer):
            self.c_sib_optimizer = CSIBOptimizerInt(n_clusters, n_features)
        else:
            self.c_sib_optimizer = CSIBOptimizerFloat(n_clusters, n_features)
        self.n_samples = n_samples
        self.xy = xy
        self.xy_sum = xy_sum
        self.x_sum = x_sum
        self.sparse = issparse(xy)

    def optimize(self, x_permutation, t_size, t_sum, t_log_sum, t_centroid, labels, ity):
        if self.sparse:
            xy_indices = self.xy.indices
            xy_indptr = self.xy.indptr
            xy_data = self.xy.data
        else:
            xy_indices = None
            xy_indptr = None
            xy_data = self.xy.ravel()
        return self.c_sib_optimizer.optimize(self.n_samples, xy_indices,
                                             xy_indptr, xy_data, self.xy_sum,
                                             self.x_sum, x_permutation, t_size, t_sum,
                                             t_log_sum, t_centroid, labels, ity)

    def infer(self, n_samples, xy, xy_sum, x_sum, t_size, t_sum, t_log_sum, t_centroid, labels, costs):
        if self.sparse:
            xy_indices = xy.indices
            xy_indptr = xy.indptr
            xy_data = xy.data
        else:
            xy_indices = None
            xy_indptr = None
            xy_data = xy.ravel()
        return self.c_sib_optimizer.infer(n_samples, xy_indices, xy_indptr, xy_data, xy_sum, x_sum,
                                          t_size, t_sum, t_log_sum, t_centroid, labels, costs)
