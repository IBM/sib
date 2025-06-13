# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.sparse import issparse
from scipy.special import xlogy


class PSIBOptimizer:

    def __init__(self, n_clusters, n_features, n_samples, xy, xy_sum, x_sum):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.xy = xy
        self.xy_sum = xy_sum
        self.x_sum = x_sum
        self.sparse = issparse(xy)

    def init_centroids(self, labels, x_ignore, t_size, t_sum, t_log_sum, t_centroid,
                       t_centroid_log_t_centroid, t_centroid_log_t_centroid_sum):
        for i in range(self.n_samples):
            if x_ignore[i]:
                continue
            t = labels[i]
            t_size[t] += 1
            t_sum[t] += self.x_sum[i]
            if self.sparse:
                i_start = self.xy.indptr[i]
                i_end = self.xy.indptr[i + 1]
                v_indices = self.xy.indices[i_start:i_end]
                v_data = self.xy.data[i_start:i_end]
                t_centroid[t, v_indices] += v_data
            else:
                t_centroid[t, :] += self.xy[i, :]
        np.log2(t_sum, out=t_log_sum)
        t_centroid_log_t_centroid[:] = xlogy(t_centroid, t_centroid) / np.log(2)
        if not self.sparse:
            t_centroid_log_t_centroid_sum[:] = np.sum(t_centroid_log_t_centroid, axis=1)

    def optimize(self, x_permutation, t_size, t_sum, t_log_sum, t_centroid,
                 t_centroid_log_t_centroid, t_centroid_log_t_centroid_sum,
                 labels, x_locked_in, ity, ref_labels=None):
        return self.iterate(True, self.n_samples, self.xy, self.xy_sum, self.x_sum,
                            x_permutation, t_size, t_sum, t_log_sum, t_centroid,
                            t_centroid_log_t_centroid, t_centroid_log_t_centroid_sum,
                            labels, x_locked_in, None, ity, ref_labels)

    def infer(self, n_samples, xy, xy_sum, x_sum, t_size, t_sum, t_log_sum, t_centroid,
              t_centroid_log_t_centroid, t_centroid_log_t_centroid_sum,
              labels, x_locked_in, costs, ref_labels=None):
        return self.iterate(False, n_samples, xy, xy_sum, x_sum, None, t_size,
                            t_sum, t_log_sum, t_centroid, t_centroid_log_t_centroid,
                            t_centroid_log_t_centroid_sum, labels, x_locked_in,
                            costs, None, ref_labels)

    def iterate(self, clustering_mode, n_samples, xy, xy_sum, x_sum,
                x_permutation, t_size, t_sum, log_t_sum, t_centroid,
                t_centroid_log_t_centroid, t_centroid_log_t_centroid_sum,
                labels, x_locked_in, costs, ity, ref_labels=None):

        n_changes = 0

        total_cost = 0

        if self.sparse:
            # log_t_sum_plus_x_sum = None
            log_t_centroid_plus_x = None
            # log_t_centroid = None
        else:
            # log_t_sum_plus_x_sum = np.empty_like(t_sum, dtype=float)
            log_t_centroid_plus_x = np.empty_like(t_centroid, dtype=float)
            # log_t_centroid = np.zeros_like(t_centroid, dtype=float)
            # np.log2(t_centroid, where=t_centroid > 0, out=log_t_centroid)

        for i in range(n_samples):
            x = x_permutation[i] if x_permutation is not None else i

            # if this element is already locked-in, we skip it
            if x_locked_in[x]:
                continue

            old_t = labels[x]

            # if t is a singleton cluster we do not reduce it any further
            if clustering_mode and t_size[old_t] == 1:
                continue

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
                # draw x out of its current cluster
                t_size[old_t] -= 1
                t_sum[old_t] -= x_sum_x
                log_t_sum[old_t] = np.log2(t_sum[old_t])
                if self.sparse:
                    t_centroid[old_t, x_indices] -= x_data
                    t_centroid_log_t_centroid[old_t, x_indices] = xlogy(np.clip(t_centroid[old_t, x_indices],
                                                                                a_min=0, a_max=None),
                                                                        t_centroid[old_t, x_indices]) / np.log(2)
                else:
                    t_centroid[old_t, :] -= x_data
                    t_centroid_log_t_centroid_sum[old_t] = np.sum(xlogy(np.clip(t_centroid[old_t, :],
                                                                                a_min=0, a_max=None),
                                                                        t_centroid[old_t, :])) / np.log(2)

            if self.sparse:
                t_centroid_log_t_centroid_sum = np.sum(t_centroid_log_t_centroid[:, x_indices], axis=1)

            # argmin Df
            t_sum_plus_x_sum = t_sum + x_sum_x

            if self.sparse:
                t_centroid_plus_x = t_centroid[:, x_indices] + x_data
                h_m_plus_t = (t_centroid_plus_x[:, None, :] @
                              np.log2(t_centroid_plus_x)[..., None]).ravel()

            else:
                t_centroid_plus_x = t_centroid + x_data
                h_m_plus_t = np.sum(xlogy(np.clip(t_centroid_plus_x, a_min=0, a_max=None),
                                          t_centroid_plus_x), axis=1) / np.log(2)

            h_t = t_centroid_log_t_centroid_sum

            tmp_costs = -h_m_plus_t + h_t - t_sum * log_t_sum + t_sum_plus_x_sum * np.log2(t_sum_plus_x_sum)
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
                log_t_sum[new_t] = np.log2(t_sum[new_t])

                if self.sparse:
                    t_centroid[new_t, x_indices] += x_data

                    t_centroid_new_t_x = t_centroid[new_t, x_indices]
                    t_centroid_log_t_centroid[new_t, x_indices] = xlogy(t_centroid_new_t_x,
                                                                        t_centroid_new_t_x) / np.log(2)
                    # no need to update t_centroid_log_t_centroid_sum because in the sparse case it is computed
                    # per x, and will be re-computed for the next x when the for-loop iterates
                else:
                    t_centroid_new_t = t_centroid[new_t, :]
                    t_centroid_new_t += x_data

                    t_centroid_log_t_centroid_sum[new_t] = np.sum(xlogy(np.clip(t_centroid_new_t, a_min=0, a_max=None),
                                                                        t_centroid_new_t)) / np.log(2)

                if new_t != old_t:
                    # update the changes counter
                    n_changes += 1
            else:
                total_cost += tmp_costs[new_t]
                costs[x, :] = tmp_costs

            labels[x] = new_t

        if clustering_mode:
            log_sum_xy = np.log2(xy_sum)
            ht = -np.dot(t_sum, log_t_sum - log_sum_xy) / xy_sum
            return n_changes / self.n_samples if self.n_samples > 0 else 0, ity, ht
        else:
            return total_cost
