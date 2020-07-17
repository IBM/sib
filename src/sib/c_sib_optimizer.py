# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from sib.c_sib_optimizer_sprase import CSIBOptimizerSparse


class CSIBOptimizer:
    def __init__(self, n_samples, n_clusters, n_features, py_x, pyx, py_x_kl, px, inv_beta):
        self.c_sib_optimizer = CSIBOptimizerSparse(
            n_samples, n_clusters, n_features,
            py_x.indices, py_x.indptr, py_x.data,
            pyx.indices, pyx.indptr, pyx.data, py_x_kl,
            px, inv_beta)

    def run(self, x_permutation, pt_x, pt, t_size, pyx_sum, ity):
        return self.c_sib_optimizer.run(x_permutation, pt_x, pt, t_size, pyx_sum, ity)

    def calc_labels_costs_score(self, pt, pyx_sum, n_samples, py_x, labels, costs, infer_mode):
        return self.c_sib_optimizer.calc_labels_costs_score(pt, pyx_sum,
                                                            n_samples, py_x.indices,
                                                            py_x.indptr, py_x.data,
                                                            labels, costs, infer_mode)

    def sparse_js(self, p_indices, p_values, q, pi1, pi2):
        return self.c_sib_optimizer.sparse_js(p_indices, p_values, q, pi1, pi2)
