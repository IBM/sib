# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# cython: language_level=3, boundscheck=False

cdef extern from "sib_optimizer_dense.cpp":
    pass

from libcpp cimport bool


# Declare the class with cdef
cdef extern from "sib_optimizer_dense.h":
    cdef cppclass SIBOptimizerDense:
        SIBOptimizerDense() except +
        SIBOptimizerDense(int n_samples, int n_clusters, int n_features,
                          const double* py_x, const double* pyx,
                          const double* py_x_kl, const double* px,
                          double inv_beta) except +

        double run(int* x_perumutation, int* pt_x, double* pt, int* t_size,
                   double* pyx_sum, double* py_t, double* ity, double *ht);

        double calc_labels_costs_score(const double* pt, const double* pyx_sum, double* py_t,
                                       int n_samples, const double* py_x, int* labels,
                                       double* costs, bool infer_mode);
