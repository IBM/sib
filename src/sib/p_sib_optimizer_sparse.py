# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from numpy import errstate, isneginf


class PSIBOptimizerSparse:

    def __init__(self, n_clusters, n_features, n_samples, xy, xy_sum, x_sum):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.xy = xy
        self.xy_sum = xy_sum
        self.x_sum = x_sum

    def optimize(self, x_permutation, t_size, t_sum, t_log_sum, t_cent_sum,
                 labels, ity, ref_labels=None):
        return self.iterate(True, self.n_samples, self.xy, self.xy_sum, self.x_sum,
                            x_permutation, t_size, t_sum, t_log_sum, t_cent_sum,
                            labels, None, ity, ref_labels)

    def infer(self, n_samples, xy, xy_sum, x_sum, t_size, t_sum, t_log_sum,
              t_cent_sum, labels, costs, ref_labels=None):
        return self.iterate(False, n_samples, xy, xy_sum, x_sum, None, t_size,
                            t_sum, t_log_sum, t_cent_sum, labels, costs, None,
                            ref_labels)

    def iterate(self, clustering_mode, n_samples, xy, xy_sum, x_sum,
                x_permutation, t_size, t_sum, t_log_sum, t_cent_sum,
                labels, costs, ity, ref_labels=None):

        n_changes = 0

        total_cost = 0

        for i in range(n_samples):
            x = x_permutation[i] if x_permutation is not None else i
            old_t = labels[x]

            if clustering_mode and t_size[old_t] == 1:
                continue  # if t is a singleton cluster we do not reduce it any further

            # obtain local references
            x_start = xy.indptr[x]
            x_end = xy.indptr[x + 1]
            x_indices = xy.indices[x_start:x_end]
            x_data = xy.data[x_start:x_end]
            x_sum_x = x_sum[x]

            if clustering_mode:
                # withdraw x from its current cluster
                t_size[old_t] -= 1
                t_sum[old_t] -= x_sum_x
                t_log_sum[old_t] = np.log2(t_sum[old_t])
                if np.issubdtype(x_data.dtype, np.integer):
                    t_cent_sum[old_t, x_indices] -= x_data
                else:
                    t_cent_sum[old_t, x_indices] = np.clip(t_cent_sum[old_t, x_indices] - x_data, a_min=0.0, a_max=None)

            log_sum_x_t = np.log2(t_sum + x_sum_x)
            t_cent_sum_x = t_cent_sum[:, x_indices]
            sum_x_cent = x_data + t_cent_sum_x
            log_sum_x_cent = np.log2(sum_x_cent)
            with errstate(divide='ignore'):
                log_cent_sum_t_x = np.log2(t_cent_sum_x)
                log_cent_sum_t_x[isneginf(log_cent_sum_t_x)] = 0
            sum1 = np.einsum('ij,ij->i', sum_x_cent, log_sum_x_t[:, None] - log_sum_x_cent)
            sum2 = np.einsum('ij,ij->i', t_cent_sum_x,  log_cent_sum_t_x - log_sum_x_t[:, None])
            tmp_costs = sum1 + sum2 + t_sum * (log_sum_x_t - t_log_sum)
            tmp_costs /= xy_sum

            new_t = np.argmin(tmp_costs).item()
            if ref_labels is not None:
                ref_t = ref_labels[x]
                if new_t != ref_t and not np.isclose(tmp_costs[new_t], tmp_costs[ref_t]):
                    print("override t of cost=%.8f, with cost=%.8f" % (tmp_costs[new_t], tmp_costs[ref_t]))
                    new_t = ref_t

            if clustering_mode:
                ity += tmp_costs[old_t] - tmp_costs[new_t]

                # add x to its new cluster
                t_size[new_t] += 1
                t_sum[new_t] += x_sum_x
                t_log_sum[new_t] = np.log2(t_sum[new_t])
                t_cent_sum[new_t, x_indices] += x_data

                if new_t != old_t:
                    # update the changes counter
                    n_changes += 1
            else:
                total_cost += tmp_costs[new_t]
                costs[x, :] = tmp_costs

            labels[x] = new_t

        if clustering_mode:
            log_sum_xy = np.log2(xy_sum)
            ht = -np.dot(t_sum, t_log_sum - log_sum_xy) / xy_sum
            return n_changes / self.n_samples if self.n_samples > 0 else 0, ity, ht
        else:
            return total_cost
