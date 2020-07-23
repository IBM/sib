/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SIB_OPTIMIZER_DENSE_H
#define SIB_OPTIMIZER_DENSE_H

class SIBOptimizerDense {
    public:
        int n_samples;
        int n_clusters;
        int n_features;

        const double* py_x;
        const double* pyx;

        const double* py_x_kl;

        const double *px;

        double inv_beta;
        bool use_inv_beta;

        SIBOptimizerDense();
        SIBOptimizerDense(int n_samples, int n_clusters, int n_features,
                          const double* py_x, const double* pyx,
                          const double* py_x_kl, const double* px, double inv_beta);
        ~SIBOptimizerDense();

        double run(int* x_permutation, int* pt_x, double* pt, int* t_size,
                   double* pyx_sum, double* py_t, double* ity, double *ht);

        double calc_labels_costs_score(const double* pt, const double* pyx_sum, double* py_t, int n_samples,
                                       const double* py_x, int* labels, double* costs, bool infer_mode);

    private:
        int calc_merge_costs(int x, double px, double* pt, double* pyx_sum, bool uniform, double* costs_x);
};

#endif
