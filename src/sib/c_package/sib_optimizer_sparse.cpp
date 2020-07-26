/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_optimizer_sparse.h"

#include <cmath>
#include <iostream>

// Overloaded constructor
SIBOptimizerSparse::SIBOptimizerSparse(int n_samples, int n_clusters, int n_features,
                                       const int* csr_indices, const int* csr_indptr,
                                       const double* py_x_data, const double* pyx_data,
                                       const double* py_x_kl, const double* px, double inv_beta) {
    this->n_samples = n_samples;
    this->n_clusters = n_clusters;
    this->n_features = n_features;
    this->csr_indices = csr_indices;
    this->csr_indptr = csr_indptr;
    this->py_x_data = py_x_data;
    this->pyx_data = pyx_data;
    this->py_x_kl = py_x_kl;
    this->px = px;
    this->inv_beta = inv_beta;
    this->use_inv_beta = inv_beta > 0.0001;
}

// Destructor
SIBOptimizerSparse::~SIBOptimizerSparse() {}

double SIBOptimizerSparse::run(int* x_permutation, int* pt_x, double* pt, int* t_size,
                               double* pyx_sum, double* ity, double *ht) {
    int n_changes = 0;

    for (int i=0; i<this->n_samples ; i++) {
        int x = x_permutation[i];
        int old_t = pt_x[x];

        // if old_t is a singleton cluster we do not reduce it any further
        if (t_size[old_t] == 1)
            continue;

        // the probability of x
        double px = this->px[x];

        // find the starting and ending indices and size
        int index_start = this->csr_indptr[x];
        int index_end = this->csr_indptr[x + 1];
        int x_size = index_end - index_start;

        // obtain pointers to indices and data
        const int* indices = &(this->csr_indices[index_start]);
        const double* py_x_data = &(this->py_x_data[index_start]);
        const double* pyx_data = &(this->pyx_data[index_start]);

        // ----------- step 1 -  draw x out of its current cluster - old_t
        // update the pt and t_size arrays
        pt[old_t] = fmax(0, pt[old_t] - px);
        t_size[old_t] -= 1;
        // update the pyx_sum array
        double* pyx_sum_t = &pyx_sum[this->n_features * old_t];
        for (int j=0 ; j<x_size ; j++) {
            int index = indices[j];
            double pyx_value = pyx_data[j];
            pyx_sum_t[index] = fmax(0, pyx_sum_t[index] - pyx_value);
        }

        // ----------- step 2 -  calculate the merge costs and find new_t     ---------
        // get the part of KL1 that relies only on py_x
        double py_x_kl1 = this->py_x_kl[x];
        // loop over the centroids and find the one to which we can add x with the minimal increase in cost
        double min_delta = 0;
        int min_delta_t = -1;
        double old_t_delta = 0;
        for (int t=0 ; t<this->n_clusters ; t++) {
            double p_new = px + pt[t];
            double pi1 = px / p_new;
            double pi2 = 1 - pi1;
            double* pyx_sum_t = &pyx_sum[this->n_features * t];
            double kl1 = py_x_kl1;
            double kl2_comp1 = 0;
            double py_t_sum = 0;
            double inv_pt_t = 1.0/pt[t];
            for (int j=0 ; j<x_size ; j++) {
                double py_x_value_j = py_x_data[j];
                double py_t_value_j = pyx_sum_t[indices[j]] * inv_pt_t;
                double average_j = py_x_value_j * pi1 + py_t_value_j * pi2;
                double log2_inv_average_j = -log2(average_j);
                kl1 += py_x_value_j * log2_inv_average_j;
                if (py_t_value_j>0) {
                    kl2_comp1 += py_t_value_j * (log2(py_t_value_j) + log2_inv_average_j);
                }
                py_t_sum += py_t_value_j;
            }
            double log2_pi2 = log2(pi2);
            double kl2_comp2 = -log2_pi2 * (1.0 - py_t_sum);
            double kl2 = kl2_comp1 + kl2_comp2;
            double js = pi1 * kl1 + pi2 * kl2;
            if (this->use_inv_beta) {
                double ent = pi1 * log2(pi1) + pi2 * log2_pi2;
                js += this->inv_beta * ent;
            }
            double delta = p_new * js;

            if (min_delta_t == -1 || delta < min_delta) {
                min_delta_t = t;
                min_delta = delta;
            }

            if (old_t == t) {
                old_t_delta = delta;
            }
        }
        int new_t = min_delta_t;
        *ity += old_t_delta - min_delta;


        // ----------- step 3 - add x to its new cluster - t_new
        // update the pt and t_size arrays
        pt[new_t] += px;
        t_size[new_t] += 1;
        // update the pyx_sum array
        double* pyx_sum_new_t = &pyx_sum[this->n_features * new_t];
        for (int j=0 ; j<x_size ; j++) {
            pyx_sum_new_t[indices[j]] += pyx_data[j];
        }

        // update pt_x and counter if switched to a new cluster
        if (new_t != old_t) {
            pt_x[x] = new_t;
            n_changes += 1;
        }
    }

    // update ht according to the updated pt
    double ht_sum = 0.0;
    for (int t=0 ; t<this->n_clusters ; t++) {
        double pt_t = pt[t];
        ht_sum += pt_t * log2(pt_t);
    }
    *ht = -ht_sum;

    return this->n_samples > 0 ? n_changes / (double)(this->n_samples) : 0;
}

double SIBOptimizerSparse::calc_labels_costs_score(const double* pt, const double* pyx_sum, int new_n_samples,
                                                   const int* new_py_x_indices, const int* new_py_x_indptr,
                                                   const double* new_py_x_data, int* labels, double* costs,
                                                   bool infer_mode) {
    double score = 0;

    double infer_mode_px;
    if (infer_mode) {
        // in infer mode we assume uniform prior so px is fixed at 1/N
        // but we increase by 1 since we treat every new sample as a
        // standalone candidate to be added to the clusters; so px is
        // updated under the asuumption that we now have N+1 samples.
        // note that pt does not need to be updated because checking
        // to which cluster to add an element is always done against
        // pt that represents N-1 elements. just like when we do a train
        // as part of the optimization step - we take x out of its
        // cluster, update pt accordingly, and then check to which
        // cluster to add it. So at the time of calculating the distances
        // pt stands for N-1 elements.
        infer_mode_px = 1.0/(this->n_samples + 1);
    } else {
        infer_mode_px = 0;
    }

    for (int x=0; x<new_n_samples ; x++) {
        double px = infer_mode ? infer_mode_px : this->px[x];

        // pointer to where we write the result
        double* costs_x = &costs[this->n_clusters * x];

        // find py_x starting and ending indices
        int index_start = new_py_x_indptr[x];
        int index_end = new_py_x_indptr[x+1];

        // obtain pointers to py_x indices and values
        const int* py_x_indices = &(new_py_x_indices[index_start]);
        const double* py_x_values = &(new_py_x_data[index_start]);
        int py_x_size = index_end - index_start;

        // calculate the part of KL1 that relies only on py_x
        double py_x_kl1 = 0;
        for (int i=0 ; i<py_x_size ; i++) {
            double py_x_value_i = py_x_values[i];
            py_x_kl1 += py_x_value_i * log2(py_x_value_i);
        }

        // loop over the centroids and find the delta from each one (+ which one yields the minimum)
        double min_delta = 0;
        int min_delta_t = -1;
        for (int t=0 ; t<this->n_clusters ; t++) {
            double pt_t = pt[t];
            double p_new = px + pt_t;
            double pi1 = px / p_new;
            double pi2 = 1 - pi1;
            const double* pyx_sum_t = &pyx_sum[this->n_features * t];
            double kl1 = py_x_kl1;
            double kl2_comp1 = 0;
            double py_t_sum = 0;
            double inv_pt_t = 1.0 / pt_t;
            for (int i=0 ; i<py_x_size ; i++) {
                double py_x_value_i = py_x_values[i];
                double py_t_value_i = pyx_sum_t[py_x_indices[i]] * inv_pt_t;
                double average_i = py_x_value_i * pi1 + py_t_value_i * pi2;
                double log2_inv_average_i = -log2(average_i);
                kl1 += py_x_value_i * log2_inv_average_i;
                if (py_t_value_i>0) {
                    kl2_comp1 += py_t_value_i * (log2(py_t_value_i) + log2_inv_average_i);
                }
                py_t_sum += py_t_value_i;
            }
            double log2_pi2 = log2(pi2);
            double kl2_comp2 = -log2_pi2 * (1.0 - py_t_sum);
            double kl2 = kl2_comp1 + kl2_comp2;
            double js = pi1 * kl1 + pi2 * kl2;
            if (this->use_inv_beta) {
                double ent = pi1 * log2(pi1) + pi2 * log2_pi2;
                js += this->inv_beta * ent;
            }
            double delta = p_new * js;
            costs_x[t] = delta;
            if (min_delta_t == -1 || delta < min_delta) {
                min_delta_t = t;
                min_delta = delta;
            }
        }
        labels[x] = min_delta_t;
        score += min_delta;
    }

    return score;
}
