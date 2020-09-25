# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# distutils: language = c++
# cython: language_level=3

from .c_sib_optimizer_sparse cimport SIBOptimizerSparse

from libc.stdint cimport int32_t, int64_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class CSIBOptimizerSparse:
    cdef SIBOptimizerSparse* c_sib_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int32_t n_clusters, int32_t n_features):
        self.c_sib_optimizer = new SIBOptimizerSparse(n_clusters, n_features)

    def __dealloc__(self):
        del self.c_sib_optimizer

    def optimize(self, int32_t n_samples, const int32_t[::1] xy_indices,
                 const int32_t[::1] xy_indptr, const int64_t[::1] xy_data,
                 int64_t sum_xy, const int64_t[::1] sum_x,
                 int32_t[::1] x_permutation,
                 int32_t[::1] t_size, int64_t[::1] sum_t,
                 int64_t[:,::1] cent_sum_t, int32_t[::1] labels,
                 double ity):
        cdef double ht = 0
        cdef double change_rate = 0
        self.c_sib_optimizer.iterate(True, n_samples, &xy_indices[0],
                                     &xy_indptr[0], &xy_data[0],
                                     sum_xy, &sum_x[0],
                                     &x_permutation[0],
                                     &t_size[0], &sum_t[0],
                                     &cent_sum_t[0, 0],
                                     &labels[0],
                                     NULL, NULL,  # costs and total cost
                                     &ity, &ht, &change_rate)
        return change_rate, ity, ht

    def infer(self, int32_t n_samples, const int32_t[::1] xy_indices,
              const int32_t[::1] xy_indptr, const int64_t[::1] xy_data,
              int64_t sum_xy, const int64_t[::1] sum_x,
              int32_t[::1] t_size, int64_t[::1] sum_t,
              int64_t[:,::1] cent_sum_t,
              int32_t[::1] labels, double[:,::1] costs):
        cdef double total_cost
        self.c_sib_optimizer.iterate(False, n_samples, &xy_indices[0],
                                     &xy_indptr[0], &xy_data[0],
                                     sum_xy, &sum_x[0],
                                     NULL,  # permutation
                                     &t_size[0], &sum_t[0],
                                     &cent_sum_t[0, 0],
                                     &labels[0], &costs[0, 0],
                                     &total_cost,
                                     NULL, NULL, NULL) # ity, ht and change_rate
        return total_cost
