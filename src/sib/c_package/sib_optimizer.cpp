/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_optimizer.h"
#include <cmath>
// #include <ctgmath>

// Constructor
template <typename T, typename FLOAT_T>
SIBOptimizer<T, FLOAT_T>::SIBOptimizer(int32_t n_clusters, int32_t n_features)
   : n_clusters(n_clusters), n_features(n_features) {}

// Destructor
template <typename T, typename FLOAT_T>
SIBOptimizer<T, FLOAT_T>::~SIBOptimizer() {}

template <typename T, typename FLOAT_T>
void SIBOptimizer<T, FLOAT_T>::init_centroids(
        int32_t n_samples, const int32_t *xy_indices,
        const int32_t *xy_indptr, const T *xy_data,
        const T* x_sum, int32_t *labels,
        int32_t *t_size, T *t_sum, FLOAT_T *t_log_sum, T *t_centroid) {

    int32_t x_start = 0;
    int32_t x_end = n_features;
    int32_t x_size = x_end - x_start;
    const int32_t* x_indices;
    const T* x_data;

    bool sparse = xy_indices != NULL;

    for (int32_t x=0; x<n_samples ; x++) {
        int32_t t = labels[x];
        t_size[t]++;
        t_sum[t] += x_sum[x];

        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &(xy_indices[x_start]);
            x_data = &(xy_data[x_start]);
        } else {
            x_data = &(xy_data[x * n_features]);
        }

        // update t_centroid
        T *t_centroid_t = &t_centroid[t * n_features];
        if (sparse) {
            for (int32_t j=0 ; j<x_size ; j++) {
                t_centroid_t[x_indices[j]] += x_data[j];
            }
        } else {
            for (int32_t j=0 ; j<x_size ; j++) {
                t_centroid_t[j] += x_data[j];
            }
        }
    }

    // set t_log_sum
    for (int32_t t=0; t<n_clusters ; t++) {
        t_log_sum[t] = log2(t_sum[t]);
    }

}


// sIB iteration over n samples for clustering / classification.
template <typename T, typename FLOAT_T>
void SIBOptimizer<T, FLOAT_T>::iterate(bool clustering_mode,// clustering / classification mode
        int32_t n_samples, const int32_t *xy_indices,       // data to cluster / classify
        const int32_t *xy_indptr, const T *xy_data,
        const T xy_sum, const T *x_sum,
        int32_t* x_permutation,                             // order of iteration
        int32_t *t_size, T *t_sum,                          // current clusters
        FLOAT_T *t_log_sum, T *t_centroid,
        int32_t *labels, FLOAT_T* costs, FLOAT_T* total_cost,  // assigned labels and costs
        FLOAT_T* ity, FLOAT_T* ht, FLOAT_T* change_rate) {     // stats on updates

    int32_t n_changes = 0;

    if (!clustering_mode) {
        *total_cost = 0;
    }

    int32_t x_start = 0;
    int32_t x_end = n_features;
    int32_t x_size = x_end - x_start;
    const int32_t* x_indices;
    const T* x_data;

    bool sparse = xy_indices != NULL;

    for (int32_t i=0; i<n_samples ; i++) {
        int32_t x = clustering_mode ? x_permutation[i] : i;
        int32_t old_t = labels[x];

        if (clustering_mode && t_size[old_t] == 1) {
            // skip elements from singleton clusters
            continue;
        }

        // obtain local pointers

        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &(xy_indices[x_start]);
            x_data = &(xy_data[x_start]);
        } else {
            x_data = &(xy_data[x * n_features]);
        }


        T x_sum_x = x_sum[x];

        if (clustering_mode) {
            // withdraw x from its current cluster
            t_size[old_t]--;
            t_sum[old_t] -= x_sum_x;
            t_log_sum[old_t] = log2(t_sum[old_t]);
            T *old_t_centroid = &(t_centroid[n_features * old_t]);
            if (sparse) {
                for (int32_t j=0 ; j<x_size ; j++) {
                    old_t_centroid[x_indices[j]] -= x_data[j];
                }
            } else {
                for (int32_t j=0 ; j<x_size ; j++) {
                    old_t_centroid[j] -= x_data[j];
                }
            }
        }

        // pointer to the costs array (used only for classification)
        FLOAT_T* x_costs = clustering_mode ? NULL : &costs[this->n_clusters * x];

        FLOAT_T min_cost = 0;
        int32_t min_cost_t = -1;
        FLOAT_T cost_old_t = 0;

        for (int32_t t=0 ; t<this->n_clusters ; t++) {
            T *t_centroid_t = &(t_centroid[n_features * t]);
            T t_sum_t = t_sum[t];
            FLOAT_T log_x_sum_plus_t_sum = log2(x_sum_x+t_sum_t);
            FLOAT_T t_log_sum_t = t_log_sum[t];
            FLOAT_T sum1 = 0;
            FLOAT_T sum2 = 0;
            FLOAT_T cost = 0;
            if (sparse) {
                for (int32_t j=0 ; j<x_size ; j++) {
                    T t_centroid_t_j = t_centroid_t[x_indices[j]];
                    T x_data_j = x_data[j];
                    T t_centroid_plus_x_j = x_data_j + t_centroid_t_j;
                    FLOAT_T log_t_centroid_plus_x_j = log2(t_centroid_plus_x_j);
                    sum1 += t_centroid_plus_x_j * (log_x_sum_plus_t_sum - log_t_centroid_plus_x_j);
                    if (t_centroid_t_j > 0) {
                        FLOAT_T log_t_centroid_t_j = log2(t_centroid_t_j);
                        sum2 += t_centroid_t_j*(log_t_centroid_t_j-log_x_sum_plus_t_sum);
                    }
                }
                cost = sum1 + sum2 + t_sum_t*(log_x_sum_plus_t_sum-t_log_sum_t);
            } else {
                for (int32_t j=0 ; j<x_size ; j++) {
                    T t_centroid_t_j = t_centroid_t[j];
                    T x_data_j = x_data[j];
                    T t_centroid_plus_x_j = x_data_j + t_centroid_t_j;
                    if (t_centroid_plus_x_j > 0) {
                        FLOAT_T log_t_centroid_plus_x_j = log2(t_centroid_plus_x_j);
                        if (t_centroid_t_j > 0) {
                            FLOAT_T log_t_centroid_t_j = log2(t_centroid_t_j);
                            sum1 += t_centroid_t_j*(log_t_centroid_t_j-t_log_sum_t);
                        }
                        sum2 += t_centroid_plus_x_j * (log_t_centroid_plus_x_j - log_x_sum_plus_t_sum);
                    }
                }
                cost = sum1 - sum2;
            }
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

            // add x to its new cluster
            t_size[new_t]++;
            t_sum[new_t] += x_sum_x;
            t_log_sum[new_t] = log2(t_sum[new_t]);
            T *new_t_centroid = &(t_centroid[n_features * new_t]);
            if (sparse) {
                for (int32_t j=0 ; j<x_size ; j++) {
                    new_t_centroid[x_indices[j]] += x_data[j];
                }
            } else {
                for (int32_t j=0 ; j<x_size ; j++) {
                    new_t_centroid[j] += x_data[j];
                }
            }

            if (new_t != old_t) {
                // update the changes counter
                n_changes++;

                // count the increase in information
                *ity += cost_old_t - min_cost;
            }

        } else {
            *total_cost += min_cost;
        }

        labels[x] = new_t;
    }

    if (clustering_mode) {
        // calculate the change rate
        *change_rate = n_samples > 0 ? n_changes / (FLOAT_T)n_samples : 0;

        // calculate the entropy of the clustering analysis
        FLOAT_T ht_sum = 0.0;
        FLOAT_T log_xy_sum = log2(xy_sum);
        for (int t=0 ; t<this->n_clusters ; t++) {
            T t_sum_t = t_sum[t];
            ht_sum += t_sum_t * (log2(t_sum_t) - log_xy_sum);
        }
        *ht = -ht_sum / (FLOAT_T)xy_sum;
    }
}
