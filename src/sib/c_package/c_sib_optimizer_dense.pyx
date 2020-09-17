# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# distutils: language = c++

# cython: language_level=3

from .c_sib_optimizer_dense cimport SIBOptimizerDense

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class CSIBOptimizerDense:
    cdef SIBOptimizerDense* c_sib_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int32_t n_clusters, int32_t n_features):
        self.c_sib_optimizer = new SIBOptimizerDense(n_clusters, n_features)

    def __dealloc__(self):
        del self.c_sib_optimizer

    def optimize(self, int32_t n_samples, const int64_t[::1] xy_data,
                 int64_t sum_xy, const int64_t[::1] sum_x,
                 int32_t[::1] x_permutation,
                 int32_t[::1] t_size, int64_t[::1] sum_t,
                 int64_t[:,::1] cent_sum_t,
                 int32_t[::1] labels, double[:,::1] costs,
                 const double[::1] log_lookup_table):
        cdef double ity_increase
        cdef double change_rate
        self.c_sib_optimizer.iterate(True, n_samples, &xy_data[0],
                                     sum_xy, &sum_x[0],
                                     &x_permutation[0],
                                     &t_size[0], &sum_t[0],
                                     &cent_sum_t[0, 0],
                                     &labels[0], &costs[0, 0],
                                     NULL, # total cost
                                     &ity_increase, &change_rate,
                                     &log_lookup_table[0])
        return change_rate, ity_increase

    def infer(self, int32_t n_samples, const int64_t[::1] xy_data,
              int64_t sum_xy, const int64_t[::1] sum_x,
              int32_t[::1] t_size, int64_t[::1] sum_t,
              int64_t[:,::1] cent_sum_t,
              int32_t[::1] labels, double[:,::1] costs,
              const double[::1] log_lookup_table):
        cdef double total_cost
        self.c_sib_optimizer.iterate(True, n_samples, &xy_data[0],
                                     sum_xy, &sum_x[0],
                                     NULL,  # permutation
                                     &t_size[0], &sum_t[0],
                                     &cent_sum_t[0, 0],
                                     &labels[0], &costs[0, 0],
                                     &total_cost,
                                     NULL, NULL, # ity and change_rate
                                     &log_lookup_table[0])
        return total_cost
