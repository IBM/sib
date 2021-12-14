# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# distutils: language = c++
# cython: language_level=3

from .c_sib_optimizer cimport SIBOptimizer

from libc.stdint cimport int32_t, int64_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)


# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods

# Python extension type.
cdef class CSIBOptimizerInt:

    cdef SIBOptimizer[int64_t]* c_sib_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int32_t n_clusters, int32_t n_features, bool fast_log):
        self.c_sib_optimizer = new SIBOptimizer[int64_t](n_clusters, n_features, fast_log)

    def __dealloc__(self):
        del self.c_sib_optimizer

    def init_centroids(self, int32_t n_samples, const int32_t[::1] xy_indices,
                       const int32_t[::1] xy_indptr, const int64_t[::1] xy_data,
                       const int64_t[::1] x_sum, int32_t[::1] labels, bool[::1] x_ignore,
                       int32_t[::1] t_size, int64_t[::1] t_sum,
                       double[::1] t_log_sum, int64_t[:,::1] t_centroid):
        self.c_sib_optimizer.init_centroids(n_samples,
                                            &xy_indices[0] if xy_indices is not None else NULL,
                                            &xy_indptr[0] if xy_indptr is not None else NULL,
                                            &xy_data[0], &x_sum[0], &labels[0], &x_ignore[0],
                                            &t_size[0], &t_sum[0], &t_log_sum[0], &t_centroid[0, 0])

    def optimize(self, int32_t n_samples, const int32_t[::1] xy_indices,
                 const int32_t[::1] xy_indptr, const int64_t[::1] xy_data,
                 int64_t xy_sum, const int64_t[::1] x_sum,
                 int32_t[::1] x_permutation,
                 int32_t[::1] t_size, int64_t[::1] t_sum,
                 double[::1] t_log_sum, int64_t[:,::1] t_centroid,
                 int32_t[::1] labels, bool[::1] x_locked_in, double ity):
        cdef double ht = 0
        cdef double change_rate = 0
        self.c_sib_optimizer.iterate(True, n_samples,
                                     &xy_indices[0] if xy_indices is not None else NULL,
                                     &xy_indptr[0] if xy_indptr is not None else NULL,
                                     &xy_data[0], xy_sum, &x_sum[0],
                                     &x_permutation[0],
                                     &t_size[0], &t_sum[0],
                                     &t_log_sum[0], &t_centroid[0, 0],
                                     &labels[0], &x_locked_in[0],
                                     NULL, NULL,  # costs and total cost
                                     &ity, &ht, &change_rate)
        return change_rate, ity, ht

    def infer(self, int32_t n_samples, const int32_t[::1] xy_indices,
              const int32_t[::1] xy_indptr, const int64_t[::1] xy_data,
              int64_t xy_sum, const int64_t[::1] x_sum,
              int32_t[::1] t_size, int64_t[::1] t_sum,
              double[::1] t_log_sum, int64_t[:,::1] t_centroid,
              int32_t[::1] labels, bool[::1] x_locked_in, double[:,::1] costs):
        cdef double total_cost
        self.c_sib_optimizer.iterate(False, n_samples,
                                     &xy_indices[0] if xy_indices is not None else NULL,
                                     &xy_indptr[0] if xy_indptr is not None else NULL,
                                     &xy_data[0], xy_sum, &x_sum[0],
                                     NULL,  # permutation
                                     &t_size[0], &t_sum[0],
                                     &t_log_sum[0], &t_centroid[0, 0],
                                     &labels[0], &x_locked_in[0],
                                     &costs[0, 0], &total_cost,
                                     NULL, NULL, NULL) # ity, ht and change_rate
        return total_cost


# Python extension type.
cdef class CSIBOptimizerFloat:

    cdef SIBOptimizer[double]* c_sib_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int32_t n_clusters, int32_t n_features, bool fast_log):
        self.c_sib_optimizer = new SIBOptimizer[double](n_clusters, n_features, fast_log)

    def __dealloc__(self):
        del self.c_sib_optimizer

    def init_centroids(self, int32_t n_samples, const int32_t[::1] xy_indices,
                       const int32_t[::1] xy_indptr, const double[::1] xy_data,
                       const double[::1] x_sum, int32_t[::1] labels, bool[::1] x_ignore,
                       int32_t[::1] t_size, double[::1] t_sum, double[::1] t_log_sum,
                       double[:,::1] t_centroid):
        self.c_sib_optimizer.init_centroids(n_samples,
                                            &xy_indices[0] if xy_indices is not None else NULL,
                                            &xy_indptr[0] if xy_indptr is not None else NULL,
                                            &xy_data[0], &x_sum[0], &labels[0], &x_ignore[0],
                                            &t_size[0], &t_sum[0], &t_log_sum[0], &t_centroid[0, 0])

    def optimize(self, int32_t n_samples, const int32_t[::1] xy_indices,
                 const int32_t[::1] xy_indptr, const double[::1] xy_data,
                 double xy_sum, const double[::1] x_sum,
                 int32_t[::1] x_permutation,
                 int32_t[::1] t_size, double[::1] t_sum,
                 double[::1] t_log_sum, double[:,::1] t_centroid,
                 int32_t[::1] labels, bool[::1] x_locked_in, double ity):
        cdef double ht = 0
        cdef double change_rate = 0
        self.c_sib_optimizer.iterate(True, n_samples,
                                     &xy_indices[0] if xy_indices is not None else NULL,
                                     &xy_indptr[0] if xy_indptr is not None else NULL,
                                     &xy_data[0], xy_sum, &x_sum[0],
                                     &x_permutation[0],
                                     &t_size[0], &t_sum[0],
                                     &t_log_sum[0], &t_centroid[0, 0],
                                     &labels[0], &x_locked_in[0], NULL, NULL,  # costs and total cost
                                     &ity, &ht, &change_rate)
        return change_rate, ity, ht

    def infer(self, int32_t n_samples, const int32_t[::1] xy_indices,
              const int32_t[::1] xy_indptr, const double[::1] xy_data,
              double xy_sum, const double[::1] x_sum,
              int32_t[::1] t_size, double[::1] t_sum,
              double[::1] t_log_sum, double[:,::1] t_centroid,
              int32_t[::1] labels, bool[::1] x_locked_in, double[:,::1] costs):
        cdef double total_cost
        self.c_sib_optimizer.iterate(False, n_samples,
                                     &xy_indices[0] if xy_indices is not None else NULL,
                                     &xy_indptr[0] if xy_indptr is not None else NULL,
                                     &xy_data[0], xy_sum, &x_sum[0],
                                     NULL,  # permutation
                                     &t_size[0], &t_sum[0],
                                     &t_log_sum[0], &t_centroid[0, 0],
                                     &labels[0], &x_locked_in[0],
                                     &costs[0, 0], &total_cost,
                                     NULL, NULL, NULL) # ity, ht and change_rate
        return total_cost
