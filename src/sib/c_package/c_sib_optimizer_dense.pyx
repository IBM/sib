# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# distutils: language = c++

# cython: language_level=3

from .c_sib_optimizer_dense cimport SIBOptimizerDense

from libcpp cimport bool

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class CSIBOptimizerDense:
    cdef SIBOptimizerDense* c_sib_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int n_samples, int n_clusters, int n_features,
                  const double[::1,:] py_x, const double[::1,:] pyx,
                  const double[::1] py_x_kl, const double[::1] px, double inv_beta):
        self.c_sib_optimizer = new SIBOptimizerDense(
            n_samples, n_clusters, n_features,
            &py_x[0, 0], &pyx[0, 0], &py_x_kl[0], &px[0], inv_beta)

    def __dealloc__(self):
        del self.c_sib_optimizer

    def run(self, int[::1] x_perumutation, int[::1] pt_x, double[::1] pt,
            int[::1] t_size, double[::1,:] pyx_sum, double[::1,:] py_t, double ity):
        cdef double hy;
        cdef double change_rate = self.c_sib_optimizer.run(
            &x_perumutation[0], &pt_x[0], &pt[0], &t_size[0], &pyx_sum[0, 0], &py_t[0, 0], &ity, &hy)
        return change_rate, ity, hy

    def calc_labels_costs_score(self, const double[::1] pt, const double[::1,:] pyx_sum,
                                double[::1,:] py_t, int n_samples, const double[::1,:] py_x,
                                int[::1] labels, double[:,::1] costs, bool infer_mode):
        return self.c_sib_optimizer.calc_labels_costs_score(&pt[0], &pyx_sum[0, 0], &py_t[0, 0], n_samples,
                                                            &py_x[0, 0], &labels[0], &costs[0, 0], infer_mode)
