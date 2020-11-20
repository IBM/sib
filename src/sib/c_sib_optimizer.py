# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.sparse import issparse

from sib.c_package.c_sib_optimizer import \
    CSIBOptimizerIntFloat as CSIBOptimizerIntFloat, \
    CSIBOptimizerIntDouble as CSIBOptimizerIntDouble, \
    CSIBOptimizerFloat as CSIBOptimizerFloat, \
    CSIBOptimizerDouble as CSIBOptimizerDouble


class CSIBOptimizer:
    def __init__(self, n_clusters, n_features, n_samples, xy, xy_sum, x_sum, float_dtype):
        if np.issubdtype(xy_sum.dtype, np.integer):
            if float_dtype == np.float32:
                self.c_sib_optimizer = CSIBOptimizerIntFloat(n_clusters, n_features)
            else:
                self.c_sib_optimizer = CSIBOptimizerIntDouble(n_clusters, n_features)
        else:
            if float_dtype == np.float32:
                self.c_sib_optimizer = CSIBOptimizerFloat(n_clusters, n_features)
            else:
                self.c_sib_optimizer = CSIBOptimizerDouble(n_clusters, n_features)
        self.n_samples = n_samples
        self.xy = xy
        self.xy_sum = xy_sum
        self.x_sum = x_sum
        self.sparse = issparse(xy)
        self.xy_indices, self.xy_indptr, self.xy_data = self.get_data(self.xy)

    def init_centroids(self, labels, t_size, t_sum, t_log_sum, t_centroid):
        return self.c_sib_optimizer.init_centroids(self.n_samples, self.xy_indices, self.xy_indptr,
                                                   self.xy_data, self.x_sum, labels, t_size, t_sum,
                                                   t_log_sum, t_centroid)

    def optimize(self, x_permutation, t_size, t_sum, t_log_sum, t_centroid, labels, ity):
        return self.c_sib_optimizer.optimize(self.n_samples, self.xy_indices,
                                             self.xy_indptr, self.xy_data, self.xy_sum,
                                             self.x_sum, x_permutation, t_size, t_sum,
                                             t_log_sum, t_centroid, labels, ity)

    def infer(self, n_samples, xy, xy_sum, x_sum, t_size, t_sum, t_log_sum, t_centroid, labels, costs):
        xy_indices, xy_indptr, xy_data = self.get_data(xy)
        return self.c_sib_optimizer.infer(n_samples, xy_indices, xy_indptr, xy_data, xy_sum, x_sum,
                                          t_size, t_sum, t_log_sum, t_centroid, labels, costs)

    def get_data(self, xy):
        if self.sparse:
            xy_indices = xy.indices
            xy_indptr = xy.indptr
            xy_data = xy.data
        else:
            xy_indices = None
            xy_indptr = None
            xy_data = xy.ravel()
        return xy_indices, xy_indptr, xy_data
