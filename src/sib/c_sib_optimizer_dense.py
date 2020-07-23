# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from sib.c_package.c_sib_optimizer_dense import CSIBOptimizerDense as CSIBOptimizer


class CSIBOptimizerDense:
    def __init__(self, n_samples, n_clusters, n_features, py_x, pyx, py_x_kl, px, inv_beta):
        self.c_sib_optimizer = CSIBOptimizer(
            n_samples, n_clusters, n_features,
            py_x, pyx, py_x_kl, px, inv_beta)

    def run(self, x_permutation, pt_x, pt, t_size, pyx_sum, py_t, ity):
        return self.c_sib_optimizer.run(x_permutation, pt_x, pt, t_size, pyx_sum, py_t, ity)

    def calc_labels_costs_score(self, pt, pyx_sum, py_t, n_samples, py_x, labels, costs, infer_mode):
        return self.c_sib_optimizer.calc_labels_costs_score(pt, pyx_sum, py_t, n_samples,
                                                            py_x, labels, costs, infer_mode)