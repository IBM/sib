/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_clustering.h"
#include "sib_utils.h"
#include "sib_optimizer_sparse.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>

using namespace std;

SIBClustering::SIBClustering(int n_clusters, int max_iter, double tol,
        bool uniform_prior, double inv_beta, int random_seed, bool verbose) {
    this->n_clusters = n_clusters;
    this->max_iter = max_iter;
    this->tol = tol;
    this->uniform_prior = uniform_prior;
    this->inv_beta = inv_beta;
    this->random_seed = random_seed;
    this->verbose = verbose;
}

SIBClustering::~SIBClustering() {
}

void SIBClustering::create_model(int n_samples, int n_features, int* csr_indptr,
        int* csr_indices, double* csr_data, int csr_data_size, double* py_x_data,
        double* pyx_data, double* px, double* py, double* hx, double* hy,
        double* hxy, double* ixy) {

    Utils::normalize(n_samples, csr_indptr, csr_indices, csr_data, csr_data_size, py_x_data);

    // calculate pyx
    if (uniform_prior) {
        double inv_sum = 1.0 / (double) n_samples;
        for (int n=0 ; n<csr_data_size ; n++) {
            pyx_data[n] = py_x_data[n] * inv_sum;
        }
    } else {
        // sum the values over the whole data
        double sum = 0;
        for (int n=0 ; n<csr_data_size ; n++) {
            sum += csr_data[n];
        }

        // divide each entry by the sum
        double inv_sum = 1.0 / sum;
        for (int n=0 ; n<csr_data_size ; n++) {
            pyx_data[n] = csr_data[n] * inv_sum;
        }
    }

    // clean py
    memset(py, 0, n_features * sizeof(double));

    // calculate px and py
    for (int i=0 ; i<n_samples ; i++) {
        int start = csr_indptr[i];
        int end = csr_indptr[i + 1];

        double x_sum = 0.0;
        for (int j=start ; j<end; j++) {
            double pyx_data_j = pyx_data[j];
            x_sum += pyx_data_j;
            py[csr_indices[j]] += pyx_data_j;
        }
        px[i] = x_sum;
    }

    // calculate the entropy of x, y and xy, and the mutual info between x and y
    *hx = Utils::entropy(px, n_samples);
    *hy = Utils::entropy(py, n_features);
    *hxy = Utils::entropy(pyx_data, csr_data_size);
    *ixy = *hx + *hy - *hxy;
}


void SIBClustering::cluster(int n_samples, int n_features, int *csr_indptr,
        int *csr_indices, double *csr_data, int csr_data_size, double* py_x_data,
        double* pyx_data, double* px, double *py, int *initial_labels, int* labels,
        double *pt, int* t_size, double* pyx_sum, double* distances, double* hx,
        double* hy, double* hxy, double* ixy, double* ht, double* hty, double *ity) {

    bool internal_model;

    if (py_x_data == NULL) {
        internal_model = true;
        py_x_data = new double[csr_data_size];
        pyx_data = new double[csr_data_size];
        create_model(n_samples, n_features, csr_indptr, csr_indices, csr_data,
                csr_data_size, py_x_data, pyx_data, px, py, hx, hy, hxy, ixy);
    } else {
        internal_model = false;
    }

    if (verbose) {
        cout<<std::fixed<<std::setprecision(2);
        cout<<"sIB probabilistic model is ready; I(X;Y)="<<*ixy<<", H(X)="<<*hx<<", H(Y)="<<*hy<<endl;
    }

    int i;      // for iterating over samples
    int j;      // for iterating over vector indices
    int t;      // for iterating over clusters
    int l;      // for iterating over elements in clusters

    // create a random generator with the given seed
    mt19937 random_engine = std::mt19937(random_seed);

    // whether we will randomize the order of trails in each iteration
    bool random_permutation;

    // initialize the labels as a random partition
    if (initial_labels != NULL) {
        memcpy(labels, initial_labels, sizeof(int) * n_samples);
        random_permutation = false;
    } else {
        int n_big_clusters = n_samples % n_clusters;
        int base_cluster_size = n_samples / n_clusters;
        for (t=0, i=0; t<n_clusters ; ++t) {
            int cluster_size = base_cluster_size + (t<n_big_clusters ? 1 : 0);
            for (l=0 ; l<cluster_size ; ++l) {
                labels[i] = t;
                i++;
            }
        }
        Utils::shuffle(labels, n_samples, random_engine);
        random_permutation = true;
    }

    // sum the vectors in each cluster
    memset(t_size, 0, n_clusters * sizeof(int));
    memset(pt, 0, n_clusters * sizeof(double));
    memset(pyx_sum, 0, n_clusters * n_features * sizeof(double));
    for (i=0; i<n_samples ; i++) {
        int start = csr_indptr[i];
        int end = csr_indptr[i + 1];
        int t = labels[i];
        t_size[t] += 1;
        pt[t] += px[i];
        double *pyx_sum_t = &pyx_sum[t * n_features];
        for (j=start ; j<end ; j++) {
            pyx_sum_t[csr_indices[j]] += pyx_data[j];
        }
    }

    // calculate the entropy of t and y, and the mutual info between t and y
    *ht = Utils::entropy(pt, n_clusters);
    *hty = Utils::entropy_safe(pyx_sum, n_features * n_clusters);
    *ity = *ht + *hy - *hty;

    if (verbose) {
        cout<<"Initial partition is ready; I(T;Y)="<<*ity<<", H(T)="<<*ht<<endl;
    }

    SIBOptimizerSparse optimizer(n_samples, n_clusters, n_features, csr_indices,
            csr_indptr, py_x_data, csr_data_size, csr_indices, csr_indptr, pyx_data,
            csr_data_size, px, inv_beta);

    int n_iter = 0;
    double update_ratio = 0;
    int* x_permutation = new int[n_samples];
    for (bool converged = false ; !converged ;) {
        if (n_iter > 0 &&  update_ratio < tol) {
            if (verbose) {
                cout<<"sIB converged in iteration "<<n_iter<<" with update rate="<<(100*update_ratio)<<"%"<<endl;
            }
            converged = true;
        } else if (max_iter > 0 && n_iter == max_iter) {
            if (verbose) {
                cout<<"sIB reached max_iter ("<<max_iter<<"); stooped before convergence."<<endl;
            }
            converged = true;
        } else {
            for(int i=0; i<n_samples; i++){
                x_permutation[i] = i;
            }
            if (random_permutation) {
                Utils::shuffle(x_permutation, n_samples, random_engine);
            }
            update_ratio = optimizer.run(x_permutation, labels, pt, t_size, pyx_sum, ity, ht);
            n_iter++;
            if (verbose) {
                cout<<"Iteration "<<n_iter<<", I<T;Y>="<<*ity<<
                      ", H(T)="<<*ht<<
                      ", update-rate="<<100*update_ratio<<"%"<<endl;
                // for debug purposes: dump the partition data members
                // partition->dump(n_iter);
            }
        }
    }

	// final iteration, updating labels and distances without changing clusters
	optimizer.calc_labels_costs_score(pt, pyx_sum, n_samples, csr_indices, csr_indptr,
			py_x_data, labels, distances, false);

    // clean ups
    delete[] x_permutation;
    if (internal_model) {
        delete[] py_x_data;
        delete[] pyx_data;
    }

    if (verbose) {
        cout<<"sIB information measures:\n\t"
                "I(T;Y)="<<*ity<<", H(T)="<<*ht<<"\n\t"<<
                "I(T;Y)/I(X;Y)="<<*ity/(*ixy)<<"\n\t"<<
                "H(T)/H(X)="<<*ht/(*hx)<<endl;
    }

}

void SIBClustering::classify(int n_samples, int n_features, double* pt,
        double* pyx_sum, int* csr_indptr, int* csr_indices, double* csr_data,
		int csr_data_size, int* labels, double* distances, double* delta) {

    SIBOptimizerSparse optimizer(0, n_clusters, n_features, NULL,
            NULL, NULL, 0, NULL,  NULL, NULL, 0, NULL, inv_beta);

    double* py_x_data = new double[csr_data_size];
    Utils::normalize(n_samples, csr_indptr, csr_indices, csr_data, csr_data_size, py_x_data);

    *delta = optimizer.calc_labels_costs_score(pt, pyx_sum, n_samples,
            csr_indices, csr_indptr, py_x_data, labels, distances, true);

    delete[] py_x_data;
}
