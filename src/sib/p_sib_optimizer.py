# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.sparse import issparse


class PSIBOptimizer:

    def __init__(self, n_clusters, n_features, n_samples, xy, xy_sum, x_sum):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.xy = xy
        self.xy_sum = xy_sum
        self.x_sum = x_sum
        self.sparse = issparse(xy)

    def optimize(self, x_permutation, t_size, t_sum, t_log_sum, t_centroid,
                 labels, ity, ref_labels=None):
        return self.iterate(True, self.n_samples, self.xy, self.xy_sum, self.x_sum,
                            x_permutation, t_size, t_sum, t_log_sum, t_centroid,
                            labels, None, ity, ref_labels)

    def infer(self, n_samples, xy, xy_sum, x_sum, t_size, t_sum, t_log_sum,
              t_centroid, labels, costs, ref_labels=None):
        return self.iterate(False, n_samples, xy, xy_sum, x_sum, None, t_size,
                            t_sum, t_log_sum, t_centroid, labels, costs, None,
                            ref_labels)

    def iterate(self, clustering_mode, n_samples, xy, xy_sum, x_sum,
                x_permutation, t_size, t_sum, t_log_sum, t_centroid,
                labels, costs, ity, ref_labels=None):

        n_changes = 0

        total_cost = 0

        if not self.sparse:
            t_log_centroid = np.log2(t_centroid, where=t_centroid > 0, out=np.zeros_like(t_centroid, dtype=float))
            t_log_centroid_plus_x = np.empty_like(t_centroid, dtype=float)
            t_log_sum_plus_x_sum = np.empty_like(t_sum, dtype=float)
        else:
            t_log_centroid = None
            t_log_centroid_plus_x = None
            t_log_sum_plus_x_sum = None

        for i in range(n_samples):
            x = x_permutation[i] if x_permutation is not None else i
            old_t = labels[x]

            if clustering_mode and t_size[old_t] == 1:
                continue  # if t is a singleton cluster we do not reduce it any further

            # obtain local references
            if self.sparse:
                x_start = xy.indptr[x]
                x_end = xy.indptr[x + 1]
                x_indices = xy.indices[x_start:x_end]
                x_data = xy.data[x_start:x_end]
            else:
                x_indices = None
                x_data = xy[x, :]
            x_sum_x = x_sum[x]

            if clustering_mode:
                # withdraw x from its current cluster
                t_size[old_t] -= 1
                t_sum[old_t] -= x_sum_x
                t_log_sum[old_t] = np.log2(t_sum[old_t])
                if self.sparse:
                    t_centroid[old_t, x_indices] -= x_data
                else:
                    old_t_centroid = t_centroid[old_t, :]
                    old_t_centroid -= x_data
                    old_t_log_centroid = t_log_centroid[old_t, :]
                    old_t_log_centroid.fill(0)
                    np.log2(old_t_centroid, where=old_t_centroid > 0, out=old_t_log_centroid)

            if self.sparse:
                t_centroid_x = t_centroid[:, x_indices]
                t_sum_plus_x_sum = t_sum + x_sum_x
                t_centroid_plus_x = t_centroid_x + x_data
                t_log_sum_plus_x_sum = np.log2(t_sum_plus_x_sum)
                t_log_centroid_plus_x = np.log2(t_centroid_plus_x)
                log_t_centroid_x = np.log2(t_centroid_x, where=t_centroid_x > 0,
                                           out=np.zeros_like(t_centroid_x, dtype=float))
                sum1 = np.einsum('ij,ij->i', t_centroid_plus_x, t_log_sum_plus_x_sum[:, None] - t_log_centroid_plus_x)
                sum2 = np.einsum('ij,ij->i', t_centroid_x,  log_t_centroid_x - t_log_sum_plus_x_sum[:, None])
                tmp_costs = sum1 + sum2 + t_sum * (t_log_sum_plus_x_sum - t_log_sum)
                tmp_costs /= xy_sum
            else:
                t_sum_plus_x_sum = t_sum + x_sum_x
                t_centroid_plus_x = t_centroid + x_data
                t_log_centroid_plus_x.fill(0)
                t_log_sum_plus_x_sum.fill(0)
                np.log2(t_centroid_plus_x, out=t_log_centroid_plus_x, where=t_centroid_plus_x > 0)
                np.log2(t_sum_plus_x_sum, out=t_log_sum_plus_x_sum, where=t_sum_plus_x_sum > 0)
                sum1 = np.einsum('ij,ij->i', t_centroid, t_log_centroid - t_log_sum[:, None])
                sum2 = np.einsum('ij,ij->i', t_centroid_plus_x, t_log_centroid_plus_x - t_log_sum_plus_x_sum[:, None])
                tmp_costs = (sum1 - sum2) / xy_sum

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

                if self.sparse:
                    t_centroid[new_t, x_indices] += x_data
                else:
                    new_t_centroid = t_centroid[new_t, :]
                    new_t_centroid += x_data
                    new_t_log_centroid = t_log_centroid[new_t, :]
                    new_t_log_centroid.fill(0)
                    np.log2(new_t_centroid, where=new_t_centroid > 0, out=new_t_log_centroid)

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
