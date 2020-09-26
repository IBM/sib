/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_optimizer_sparse.h"
#include <cmath>


// Constructor
template <typename T>
SIBOptimizerSparse<T>::SIBOptimizerSparse(int32_t n_clusters, int32_t n_features)
   : n_clusters(n_clusters), n_features(n_features) {}

// Destructor
template <typename T>
SIBOptimizerSparse<T>::~SIBOptimizerSparse() {}

// sIB iteration over n samples for clustering / classification.
template <typename T>
void SIBOptimizerSparse<T>::iterate(bool clustering_mode,      // clustering / classification mode
        int32_t n_samples, const int32_t *xy_indices,       // data to cluster / classify
        const int32_t *xy_indptr, const T *xy_data,
        const T xy_sum, const T *x_sum,
        int32_t* x_permutation,                             // order of iteration
        int32_t *t_size, T *t_sum,                          // current clusters
        double *t_log_sum, T *t_centroid,
        int32_t *labels, double* costs, double* total_cost, // assigned labels and costs
        double* ity, double* ht, double* change_rate) {     // stats on updates

    int32_t n_changes = 0;

    if (!clustering_mode) {
        *total_cost = 0;
    }

    for (int32_t i=0; i<n_samples ; i++) {
        int32_t x = clustering_mode ? x_permutation[i] : i;
        int32_t old_t = labels[x];

        if (clustering_mode && t_size[old_t] == 1) {
            // skip elements from singleton clusters
            continue;
        }

        // obtain local pointers
        int32_t x_start = xy_indptr[x];
        int32_t x_end = xy_indptr[x + 1];
        int32_t x_size = x_end - x_start;
        const int32_t* x_indices = &(xy_indices[x_start]);
        const T* x_data = &(xy_data[x_start]);
        T x_sum_x = x_sum[x];

        if (clustering_mode) {
            // withdraw x from its current cluster
            t_size[old_t]--;
            t_sum[old_t] -= x_sum_x;
            t_log_sum[old_t] = log2(t_sum[old_t]);
            T *old_t_centroid = &(t_centroid[n_features * old_t]);
            for (int32_t j=0 ; j<x_size ; j++) {
                old_t_centroid[x_indices[j]] -= x_data[j];
            }

        }

        // pointer to the costs array (used only for classification)
        double* x_costs = clustering_mode ? NULL : &costs[this->n_clusters * x];

        double min_cost = 0;
        int32_t min_cost_t = -1;
        double cost_old_t = 0;

        for (int32_t t=0 ; t<this->n_clusters ; t++) {
            T *t_centroid_t = &(t_centroid[n_features * t]);
            T t_sum_t = t_sum[t];
            double log_x_sum_plus_t_sum = log2(x_sum_x+t_sum_t);
            double log_x_sum_plus_t_sum2 = log2(t_sum_t);
            double t_log_sum_t = t_log_sum[t];
            double sum1 = 0;
            double sum2 = 0;
            for (int32_t j=0 ; j<x_size ; j++) {
                T t_centroid_t_j = t_centroid_t[x_indices[j]];
                T x_data_j = x_data[j];
                T sum_j = x_data_j + t_centroid_t_j;
                double log_sum_j = log2(sum_j);
                sum1 += sum_j * (log_x_sum_plus_t_sum - log_sum_j);
                if (t_centroid_t_j > 0) {
                    double log_t_centroid_t_j = log2(t_centroid_t_j);
                    sum2 += t_centroid_t_j*(log_t_centroid_t_j-log_x_sum_plus_t_sum);
                }
            }
            double cost = sum1 + sum2 + t_sum_t*(log_x_sum_plus_t_sum-t_log_sum_t);
            cost /= xy_sum;

            if (min_cost_t == -1 || cost < min_cost) {
                min_cost_t = t;
                min_cost = cost;
            }

            if (clustering_mode) {
                if (t == old_t) {
                    cost_old_t = cost;
                }
            } else {
                x_costs[t] = cost;
            }
        }

        int32_t new_t = min_cost_t;

        if (clustering_mode) {
            // count the increase in information
            *ity += cost_old_t - min_cost;

            // add x to its new cluster
            t_size[new_t]++;
            t_sum[new_t] += x_sum_x;
            t_log_sum[new_t] = log2(t_sum[new_t]);
            T *new_t_centroid = &(t_centroid[n_features * new_t]);
            for (int32_t j=0 ; j<x_size ; j++) {
                new_t_centroid[x_indices[j]] += x_data[j];
            }

            if (new_t != old_t) {
                // update the changes counter
                n_changes++;
            }

        } else {
            *total_cost += min_cost;
        }

        labels[x] = new_t;
    }

    if (clustering_mode) {
        // calculate the change rate
        *change_rate = n_samples > 0 ? n_changes / (double)n_samples : 0;

        // calculate the entropy of the clustering analysis
        double ht_sum = 0.0;
        double log_xy_sum = log2(xy_sum);
        for (int t=0 ; t<this->n_clusters ; t++) {
            T t_sum_t = t_sum[t];
            ht_sum += t_sum_t * (log2(t_sum_t) - log_xy_sum);
        }
        *ht = -ht_sum / (double)xy_sum;
    }

}
