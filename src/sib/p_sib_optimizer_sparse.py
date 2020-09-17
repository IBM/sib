# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np


class PSIBOptimizerSparse:

    def __init__(self, n_clusters, n_features, n_samples, xy, sum_xy, sum_x):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.xy = xy
        self.sum_xy = sum_xy
        self.sum_x = sum_x

    def optimize(self, x_permutation, t_size, sum_t, cent_sum_t,
                 labels, ity, log_lookup_table, ref_labels=None):
        return self.iterate(True, self.n_samples, self.xy,
                            self.sum_xy, self.sum_x,
                            x_permutation,
                            t_size, sum_t, cent_sum_t,
                            labels, None, ity,  # costs
                            log_lookup_table, ref_labels)

    def infer(self, n_samples, xy, sum_xy, sum_x, t_size, sum_t,
              cent_sum_t, labels, costs, log_lookup_table, ref_labels=None):
        return self.iterate(False, n_samples, xy, sum_xy, sum_x,
                            None,  # permutation
                            t_size, sum_t, cent_sum_t,
                            labels, costs, None,  # ity
                            log_lookup_table, ref_labels)

    def iterate(self, clustering_mode, n_samples, xy, sum_xy, sum_x,
                x_permutation, t_size, sum_t, cent_sum_t,
                labels, costs, ity, log_lookup_table, ref_labels=None):

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
            sum_x_x = sum_x[x]

            if clustering_mode:
                # withdraw x from its current cluster
                t_size[old_t] -= 1
                sum_t[old_t] -= sum_x_x
                cent_sum_t[old_t, x_indices] -= x_data

            log_sum_x_t = np.take(log_lookup_table, sum_t + sum_x_x)
            log_sum_t = np.take(log_lookup_table, sum_t)
            cent_sum_t_x = cent_sum_t[:, x_indices]
            sum_x_cent = x_data + cent_sum_t_x
            log_sum_x_cent = np.take(log_lookup_table, sum_x_cent)
            log_cent_sum_t_x = np.take(log_lookup_table, cent_sum_t_x)
            sum1 = np.einsum('ij,ij->i', sum_x_cent, log_sum_x_t[:, None] - log_sum_x_cent)
            sum2 = np.einsum('ij,ij->i', cent_sum_t_x,  log_cent_sum_t_x - log_sum_x_t[:, None])
            tmp_costs = sum1 + sum2 + sum_t * (log_sum_x_t - log_sum_t)
            tmp_costs /= sum_xy

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
                sum_t[new_t] += sum_x_x
                cent_sum_t[new_t, x_indices] += x_data

                if new_t != old_t:
                    # update the changes counter
                    n_changes += 1
            else:
                total_cost += tmp_costs[new_t]
                costs[x, :] = tmp_costs

            labels[x] = new_t

        if clustering_mode:
            log_sum_xy = log_lookup_table[sum_xy]
            log_sum_t = np.take(log_lookup_table, sum_t)
            ht = -np.dot(sum_t, log_sum_t - log_sum_xy) / sum_xy
            return n_changes / self.n_samples if self.n_samples > 0 else 0, ity, ht
        else:
            return total_cost
