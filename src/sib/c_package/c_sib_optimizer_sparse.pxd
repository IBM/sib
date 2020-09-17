# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# cython: language_level=3, boundscheck=False

cdef extern from "sib_optimizer_sparse.cpp":
    pass

from libcpp cimport bool
from libc.stdint cimport int32_t, int64_t


# Declare the class with cdef
cdef extern from "sib_optimizer_sparse.h":
    cdef cppclass SIBOptimizerSparse:
        SIBOptimizerSparse(int32_t n_clusters, int32_t n_features);
        void iterate(
                bool clustering_mode,
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const int64_t *xy_data,
                int64_t sum_xy, const int64_t* sum_x,
                int32_t* x_permutation,
                int32_t *t_size, int64_t *sum_t, int64_t *cent_sum_t,
                int32_t *labels, double* costs, double* total_cost,
                double* ity, double* ht, double* change_rate,
                const double* log_lookup_table);
