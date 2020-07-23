# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from numpy import errstate, isneginf
from scipy.stats import entropy


class PSIBOptimizerSparse:
    def __init__(self, n_samples, n_clusters, n_features, py_x, pyx, py_x_kl, px, inv_beta):
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.py_x = py_x
        self.pyx = pyx
        self.py_x_kl = py_x_kl
        self.px = px
        self.inv_beta = inv_beta
        self.indexed_py_x = self.sparse_matrix_indexer(self.py_x)
        self.indexed_pyx = self.sparse_matrix_indexer(self.pyx)

    def run(self, x_permutation, pt_x, pt, t_size, pyx_sum, py_t, ity, ref_pt_x=None):
        changes_count = 0

        for j in range(self.n_samples):
            x = x_permutation[j]
            px = self.px[x]
            t = pt_x[x]

            if t_size[t] == 1:
                continue  # if t is a singleton cluster we do not reduce it any further

            # draw x out of its current cluster t
            pt[t] -= px
            pt[t] = max(0.0, pt[t])
            t_size[t] -= 1
            pyx_indices, pyx_values = self.indexed_pyx[x]
            pyx_sum_t = pyx_sum[:, t]
            pyx_sum_t[pyx_indices] = np.clip(pyx_sum_t[pyx_indices] - pyx_values, a_min=0.0, a_max=None)

            # calculate merge costs and select the new t
            new_t, delta = self.calc_merge_costs(x, px, pt, pyx_sum, False,
                                                 self.indexed_py_x, None, t,
                                                 ref_pt_x[x] if ref_pt_x is not None else None)
            ity += delta

            # update membership
            pt[new_t] += px
            pyx_sum_new_t = pyx_sum[:, new_t]
            pyx_sum_new_t[pyx_indices] += pyx_values
            t_size[new_t] += 1

            if not new_t == t:
                pt_x[x] = new_t
                changes_count += 1

        return changes_count / self.n_samples, ity, entropy(pt, base=2)

    def calc_merge_costs(self, x, px, pt, pyx_sum, uniform, indexed_py_x, out_costs, old_t, ref_t):
        # T.costs[x, t] = (px+pt(t))*JS(py_x,py_t(:, t)) - inv_beta*H([pi1 pi2])

        # definitions/calculations needed for both kl1 and kl2
        p_new = 1 if uniform else px + pt
        pi1 = np.full(self.n_clusters, 0.5) if uniform else px / p_new
        pi2 = 1 - pi1
        py_x_indices, py_x_values = indexed_py_x[x]
        py_t_values = pyx_sum[py_x_indices, :] * (1 / pt)
        # einsum: multiply pi by a tiling of py_x_values: (pi1[0]*v | pi1[1]*v | pi1[2]*v ...)
        average = np.einsum('i,j->ji', pi1, py_x_values) + pi2 * py_t_values
        log_inv_average = -np.log2(average)

        # kl1 calculation
        kl1 = self.py_x_kl[x] + py_x_values.dot(log_inv_average)

        # kl2 calculation
        with errstate(divide='ignore'):
            py_t_values_log = np.log2(py_t_values)
            py_t_values_log[isneginf(py_t_values_log)] = 0
        # multiply element-wise, and return an array of the sum of each column
        kl2_comp1 = np.einsum("ij,ij->j", py_t_values, py_t_values_log + log_inv_average)
        py_t_sum = np.sum(py_t_values, axis=0)
        log2_pi2 = np.log2(pi2)
        kl2_comp2 = -log2_pi2 * (1.0 - py_t_sum)
        kl2 = kl2_comp1 + kl2_comp2

        # calc js
        js = pi1 * kl1 + pi2 * kl2

        # update if inverse beta is applied
        if self.inv_beta:
            ent = pi1 * np.log2(pi1) + pi2 * np.log2(pi2)
            js += self.inv_beta * ent

        costs = p_new * js

        if out_costs is not None:
            np.copyto(out_costs, costs)

        new_t = np.argmin(costs).item()
        if ref_t is not None and new_t != ref_t:
            if np.isclose(costs[new_t], costs[ref_t]):
                print("override cost=%.8f, with cost=%.8f" % (costs[new_t], costs[ref_t]))
                new_t = ref_t

        if old_t is not None:
            return new_t, costs[old_t] - costs[new_t]

        return new_t

    def calc_labels_costs_score(self, pt, pyx_sum, py_t, n_samples, py_x, labels, costs, infer_mode):
        score = 0
        if infer_mode:
            px = np.full(n_samples, 1 / (self.n_samples + 1))
            indexed_py_x = self.sparse_matrix_indexer(py_x)
        else:
            px = self.px
            indexed_py_x = self.indexed_py_x
        for x in range(n_samples):
            label = self.calc_merge_costs(x=x, px=px[x], pt=pt,
                                          pyx_sum=pyx_sum, uniform=False,
                                          indexed_py_x=indexed_py_x,
                                          out_costs=costs[x],
                                          old_t=None, ref_t=None)
            labels[x] = label
            score += costs[x][label]
        return score

    @staticmethod
    def sparse_matrix_indexer(csr_matrix):
        data = csr_matrix.data
        indices = csr_matrix.indices
        indptr = csr_matrix.indptr
        row_data = np.empty(len(indptr) - 1, dtype=object)
        for i in range(len(indptr) - 1):
            indices_i = indices[indptr[i]:indptr[i + 1]]
            data_i = data[indptr[i]:indptr[i + 1]]
            row_data[i] = (indices_i, data_i)
        return row_data
