# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# cython: language_level=3, boundscheck=False

cdef extern from "sib_optimizer.cpp":
    pass

from libcpp cimport bool
from libc.stdint cimport int32_t


# Declare the class with cdef
cdef extern from "sib_optimizer.h":
    cdef cppclass SIBOptimizer[T]:
        SIBOptimizer(int32_t n_clusters, int32_t n_features, bool fast_log);

        void init_centroids(
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const T *xy_data,
                const T* x_sum, int32_t *labels, bool* x_ignore,
                int32_t *t_size, T *t_sum, double *t_log_sum, T *t_centroid);

        void iterate(
                bool clustering_mode,
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const T *xy_data,
                T xy_sum, const T* x_sum,
                int32_t* x_permutation,
                int32_t *t_size, T *t_sum, double *t_log_sum, T *t_centroid,
                int32_t *labels, bool *x_locked_in,
                double* costs, double* total_cost,
                double* ity, double* ht, double* change_rate);
