# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.stats import entropy
# from numpy import errstate, isneginf


class PSIBOptimizerDense:
    def __init__(self, n_clusters, n_features, n_samples, xy, sum_xy, sum_x):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.xy = xy
        self.sum_xy = sum_xy
        self.sum_x = sum_x

    def optimize(self, x_permutation, t_size, sum_t, cent_sum_t, labels, ity, log_lookup_table, ref_labels=None):
        return self.iterate(True, self.n_samples, self.xy,
                            self.sum_xy, self.sum_x,
                            x_permutation,
                            t_size, sum_t, cent_sum_t,
                            labels,
                            None, None,  # costs and total cost
                            log_lookup_table)

    def infer(self, n_samples, xy, sum_xy, sum_x, t_size, sum_t,
              cent_sum_t, labels, costs, log_lookup_table):
        return self.iterate(n_samples, xy, sum_xy, sum_x,
                            None,  # permutation
                            t_size, sum_t, cent_sum_t,
                            labels, costs,
                            None,  # total cost
                            log_lookup_table)

    def iterate(self, n_samples, xy, sum_xy, sum_x,
                x_permutation, t_size, sum_t, cent_sum_t,
                labels, costs, total_cost, log_lookup_table):

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
            pyx_values = self.pyx[:, x]
            np.clip(pyx_sum[:, t] - pyx_values, a_min=0.0, a_max=None, out=pyx_sum[:, t])
            py_t[:, t] = pyx_sum[:, t] / pt[t]

            # calculate merge costs and select the new t
            new_t, delta = self.calc_merge_costs(x, px, pt, py_t, self.py_x, None, t,
                                                 ref_pt_x[x] if ref_pt_x is not None else None)
            ity += delta

            # update membership
            pt[new_t] += px
            t_size[new_t] += 1
            pyx_sum[:, new_t] += pyx_values
            py_t[:, new_t] = pyx_sum[:, new_t] / pt[new_t]

            if not new_t == t:
                pt_x[x] = new_t
                changes_count += 1

        return changes_count / self.n_samples, ity, entropy(pt, base=2)

    def calc_merge_costs(self, x, px, pt, py_t, py_x, out_costs, old_t, ref_t):
        # T.costs[x, t] = (px+pt(t))*JS(py_x,py_t(:, t)) - inv_beta*H([pi1 pi2])

        # definitions/calculations needed for both kl1 and kl2
        p_new = px + pt
        pi1 = px / p_new
        pi2 = 1 - pi1

        # calc js
        py_x_tile = np.tile(py_x[:, x], (self.n_clusters, 1)).T
        average = pi1 * py_x_tile + pi2 * py_t
        kl1 = entropy(py_x_tile, average, base=2)
        kl2 = entropy(py_t, average, base=2)

        # alternative way to calc js, supposed to be faster because it avoids the 'tile' function
        # but it turned out to be slower than the above, perhaps because of the way that this code
        # handles the logs, and because it makes more calls to numpy
        # einsum: multiply pi by a tiling of py_x_values: (pi1[0]*v | pi1[1]*v | pi1[2]*v ...)
        # py_x_x = py_x[:, x]
        # average = np.einsum('i,j->ji', pi1, py_x_x) + pi2 * py_t
        # kl2 = entropy(py_t, average, base=2)
        # with errstate(divide='ignore'):
        #    np.log2(average, out=average)
        #    average[isneginf(average)] = 0
        # kl1 = self.py_x_kl[x] - np.einsum('i,ij->j', py_x_x, average)

        js = pi1 * kl1 + pi2 * kl2

        # update if inverse beta is applied
        if self.inv_beta:
            ent = pi1 * np.log(pi1) + pi2 * np.log(pi2)
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
        else:
            px = self.px
            py_x = self.py_x
        for x in range(n_samples):
            label = self.calc_merge_costs(x=x, px=px[x], pt=pt,
                                          py_t=py_t, py_x=py_x,
                                          out_costs=costs[x],
                                          old_t=None, ref_t=None)
            labels[x] = label
            score += costs[x][label]

        return score
