# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from sib.c_package.c_sib_optimizer_sprase import CSIBOptimizerSparse as CSIBOptimizer


class CSIBOptimizerSparse:
    def __init__(self, n_clusters, n_features, n_samples, xy, sum_xy, sum_x):
        self.c_sib_optimizer = CSIBOptimizer(n_clusters, n_features)
        self.n_samples = n_samples
        self.xy = xy
        self.sum_xy = sum_xy
        self.sum_x = sum_x

    def optimize(self, x_permutation, t_size, sum_t, cent_sum_t, labels, ity, log_lookup_table):
        return self.c_sib_optimizer.optimize(self.n_samples, self.xy.indices,
                                             self.xy.indptr, self.xy.data, self.sum_xy,
                                             self.sum_x, x_permutation, t_size,
                                             sum_t, cent_sum_t, labels, ity, log_lookup_table)

    def infer(self, n_samples, xy, sum_xy, sum_x, t_size, sum_t,
              cent_sum_t, labels, costs, log_lookup_table):
        return self.c_sib_optimizer.infer(n_samples, xy.indices, xy.indptr, xy.data, sum_xy,
                                          sum_x, t_size, sum_t, cent_sum_t, labels, costs,
                                          log_lookup_table)
