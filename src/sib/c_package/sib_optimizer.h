/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SIB_OPTIMIZER_H
#define SIB_OPTIMIZER_H

#include <stdint.h>


template <typename T>
class SIBOptimizer {
    public:
        SIBOptimizer(int32_t n_clusters, int32_t n_features, bool fast_log);
        virtual ~SIBOptimizer();

        void init_centroids(
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const T *xy_data,
                const T* x_sum, int32_t *labels, bool *x_ignore,
                int32_t *t_size, T *t_sum, double *t_log_sum, T *t_centroid);

        void iterate(bool clustering_mode,                          // clustering / classification mode
                int32_t n_samples, const int32_t *xy_indices,       // data to cluster / classify
                const int32_t *xy_indptr, const T *xy_data,
                const T xy_sum, const T *x_sum,
                int32_t* x_permutation,                             // order of iteration
                int32_t *t_size, T *t_sum,                          // current clusters
                double *t_log_sum, T *t_centroid,
                int32_t *labels, bool *locked_in,
                double* costs, double* total_cost, // assigned labels and costs
                double* ity, double* ht, double* change_rate);      // stats on updates

    private:
        int32_t n_clusters;
        int32_t n_features;
        double (*log2_ptr)(double);
};

#endif // SIB_OPTIMIZER_H

