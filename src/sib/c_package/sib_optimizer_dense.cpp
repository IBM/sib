/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_optimizer_dense.h"
#include <cmath>


// Overloaded constructor
SIBOptimizerDense::SIBOptimizerDense(int32_t n_clusters, int32_t n_features) {
    this->n_clusters = n_clusters;
    this->n_features = n_features;
}


// Destructor
SIBOptimizerDense::~SIBOptimizerDense() {}

// sIB iteration over n samples for clustering / classification.
void SIBOptimizerDense::iterate(
        // clustering / classification mode:
        bool clustering_mode,
        // data to cluster / classify:
        int32_t n_samples, const int64_t *xy_data, int64_t sum_xy, const int64_t* sum_x,
        // order of iteration:
        int32_t* x_permutation,
        // current clusters:
        int32_t *t_size, int64_t *sum_t, int64_t *cent_sum_t,
        // assigned labels and costs:
        int32_t *labels, double* costs, double* total_cost,
        // stats on updates:
        double* ity_increase, double* change_rate,
        // lookup table for log2
        const double* log_lookup_table) {

    int32_t n_changes = 0;

    if (clustering_mode) {
        *ity_increase = 0;
    } else {
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
        const int64_t* x_data = &xy_data[n_features * x];
        int64_t sum_x_x = sum_x[x];

        if (clustering_mode) {
            // withdraw x from its current cluster
            t_size[old_t]--;
            sum_t[old_t] -= sum_x_x;
            int64_t *cent_sum_t_old_t = &(cent_sum_t[n_features * old_t]);
            for (int32_t j=0 ; j<n_features ; j++) {
                cent_sum_t_old_t[j] -= x_data[j];
            }
        }

        // pointer to the costs array (used only for classification)
        double* x_costs = clustering_mode ? NULL : &costs[this->n_clusters * x];

        double min_cost = 0;
        int32_t min_cost_t = -1;
        double cost_old_t = 0;
        for (int32_t t=0 ; t<this->n_clusters ; t++) {
            int64_t *cent_sum_t_t = &(cent_sum_t[n_features * t]);
            int64_t sum_t_t = sum_t[t];
            // double log_sum_x_t = log2(sum_x_x+sum_t_t);
            double log_sum_x_t = log_lookup_table[sum_x_x+sum_t_t];
            //double log_sum_t_t = log_sum_t[t];
            double log_sum_t_t = log_lookup_table[sum_t_t];
            double sum = 0;
            double sum2 = 0;
            for (int32_t j=0 ; j<n_features ; j++) {
                int64_t cent_sum_t_t_j = cent_sum_t_t[j];
                int64_t x_data_j = x_data[j];
                int64_t sum_j = x_data_j + cent_sum_t_t_j;
                //double log_sum_j = log2(sum_j);
                double log_sum_j = log_lookup_table[sum_j];
                sum += sum_j * (log_sum_x_t - log_sum_j);
                // printf("log=%d\n", (int)(sum_j));
                if (cent_sum_t_t_j > 0) {
                    //double log_cent_sum_t_t_j = log2(cent_sum_t_t_j);
                    double log_cent_sum_t_t_j = log_lookup_table[cent_sum_t_t_j];
                    // printf("log=%d\n", (int)(cent_sum_t_t_j));
                    sum2 += cent_sum_t_t_j*(log_cent_sum_t_t_j-log_sum_x_t);
                }
            }
            double cost = sum + sum2 + sum_t_t*(log_sum_x_t-log_sum_t_t);
            cost /= sum_xy;

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
            *ity_increase += cost_old_t - min_cost;

            // add x to its new cluster
            t_size[old_t]++;
            sum_t[new_t] += sum_x_x;
            int64_t *cent_sum_t_new_t = &(cent_sum_t[n_features * new_t]);
            for (int32_t j=0 ; j<n_features ; j++) {
                cent_sum_t_new_t[j] += x_data[j];
            }

            if (new_t != old_t) {
                // update the changes counter
                n_changes += 1;
            }

        } else {
            *total_cost += min_cost;
        }

        labels[x] = new_t;
    }

    if (clustering_mode) {
        *change_rate = n_samples > 0 ? n_changes / (double)n_samples : 0;
    }
}
