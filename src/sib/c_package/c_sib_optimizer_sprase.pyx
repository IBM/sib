# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# distutils: language = c++
# cython: language_level=3

from c_sib_optimizer_sparse cimport SIBOptimizerSparse

from libcpp cimport bool

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class CSIBOptimizerSparse:
    cdef SIBOptimizerSparse* c_sib_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int n_samples, int n_clusters, int n_features,
                  const int[::1] py_x_indices, const int[::1] py_x_indptr, const double[::1] py_x_data,
                  const int[::1] pyx_indices, const int[::1] pyx_indptr, const double[::1] pyx_data,
                  const double[::1] py_x_kl, const double[::1] px, double inv_beta):
        self.c_sib_optimizer = new SIBOptimizerSparse(
            n_samples, n_clusters, n_features,
            &py_x_indices[0], &py_x_indptr[0], &py_x_data[0], len(py_x_data),
            &pyx_indices[0], &pyx_indptr[0], &pyx_data[0], len(pyx_data),
            &py_x_kl[0], &px[0], inv_beta)

    def __dealloc__(self):
        del self.c_sib_optimizer

    def run(self, int[::1] x_perumutation, int[::1] pt_x, double[::1] pt,
            int[::1] t_size, double[::1,:] pyx_sum, double ity):
        cdef double hy;
        cdef double change_rate = self.c_sib_optimizer.run(
            &x_perumutation[0], &pt_x[0], &pt[0], &t_size[0], &pyx_sum[0, 0], &ity, &hy)
        return change_rate, ity, hy

    def calc_labels_costs_score(self, const double[::1] pt, const double[::1,:] pyx_sum,
                                int n_samples, const int[::1] py_x_indices,
                                const int[::1] py_x_indptr, const double[::1] py_x_data,
                                int[::1] labels, double[:,::1] costs, bool infer_mode):
        return self.c_sib_optimizer.calc_labels_costs_score(&pt[0], &pyx_sum[0, 0], n_samples,
                                                            &py_x_indices[0], &py_x_indptr[0],
                                                            &py_x_data[0],
                                                            &labels[0], &costs[0, 0], infer_mode)

    def sparse_js(self, int[::1] p_indices, double[::1] p_values, double[::1] q, double pi1, double pi2):
        return self.c_sib_optimizer.sparse_js(&p_indices[0], &p_values[0], len(p_indices), &q[0], pi1, pi2)
