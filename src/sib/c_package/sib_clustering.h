/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SIB_CLUSTERING_H_
#define SIB_CLUSTERING_H_

class SIBClustering {
public:
    SIBClustering(int n_clusters, int max_iter, double tol, bool uniform_prior,
            double inv_beta, int random_seed, bool verbose);

    virtual ~SIBClustering();

    void create_model(int n_samples, int n_features, int* csr_indptr, int* csr_indices,
            double* csr_data, int csr_data_size, double* py_x_data, double* pyx_data,
            double* px, double* py, double* hx, double* hy, double* hxy, double* ixy);

    void cluster(int n_samples, int n_features, int *csr_indptr, int *csr_indices,
            double *csr_data, int csr_data_size, double* py_x_data, double* pyx_data,
            double* px, double *py, int *initial_labels, int* labels, double *pt,
            int* t_size, double* pyx_sum, double* distances, double* hx, double* hy,
            double* hxy, double* ixy, double* ht, double* hty, double* ity);

    void classify(int n_samples, int n_features, double* pt, double* pyx_sum,
    		int* csr_indptr, int* csr_indices, double* csr_data, int csr_data_size,
			int* labels, double* distances, double* delta);

private:
    int n_clusters;
    int max_iter;
    double tol;
    bool uniform_prior;
    double inv_beta;
    int random_seed;
    bool verbose;
};

#endif /* SIB_CLUSTERING_H_ */
