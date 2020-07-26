/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SIB_OPTIMIZER_SPARSE_H
#define SIB_OPTIMIZER_SPARSE_H

class SIBOptimizerSparse {
    public:
        int n_samples;
        int n_clusters;
        int n_features;

        const int* csr_indices;
        const int* csr_indptr;
        const double* py_x_data;
        const double* pyx_data;

        const double* py_x_kl;

        const double *px;

        double inv_beta;
        bool use_inv_beta;

        SIBOptimizerSparse(int n_samples, int n_clusters, int n_features,
                           const int* csr_indices, const int* csr_indptr,
                           const double* py_x_data, const double* pyx_data,
                           const double* py_x_kl, const double* px, double inv_beta);
        virtual ~SIBOptimizerSparse();

        double run(int* x_permutation, int* pt_x, double* pt, int* t_size,
                   double* pyx_sum, double* ity, double *ht);

        double calc_labels_costs_score(const double* pt, const double* pyx_sum, int n_samples,
                                       const int* py_x_indices, const int* py_x_indptr, const double* py_x_data,
                                       int* labels, double* costs, bool infer_mode);

    private:
        int calc_merge_costs(int x, double px, double* pt, double* pyx_sum, bool uniform, double* costs_x);
};

#endif
