/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_optimizer_dense.h"

#include <cmath>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

// Overloaded constructor
SIBOptimizerDense::SIBOptimizerDense(int n_samples, int n_clusters, int n_features,
                                     const double* py_x, const double* pyx,
                                     const double* py_x_kl, const double* px, double inv_beta) {
    this->n_samples = n_samples;
    this->n_clusters = n_clusters;
    this->n_features = n_features;
    this->py_x = py_x;
    this->pyx = pyx;
    this->py_x_kl = py_x_kl;
    this->px = px;
    this->inv_beta = inv_beta;
    this->use_inv_beta = inv_beta > 0.0001;
}

// Destructor
SIBOptimizerDense::~SIBOptimizerDense () {}

double SIBOptimizerDense::run(int* x_permutation, int* pt_x, double* pt, int* t_size,
                              double* pyx_sum, double* py_t, double* ity, double *ht) {
    int n_changes = 0;

    for (int i=0; i<this->n_samples ; i++) {
        int x = x_permutation[i];
        int old_t = pt_x[x];

        // if old_t is a singleton cluster we do not reduce it any further
        if (t_size[old_t] == 1)
            continue;

        double px = this->px[x];
        const double* pyx_x = &pyx[n_features * x];

        // ----------- step 1 -  draw x out of its current cluster - old_t
        // update the pt and t_size arrays
        pt[old_t] = fmax(0, pt[old_t] - px);
        t_size[old_t] -= 1;
        // update the pyx_sum array
        double* pyx_sum_t = &pyx_sum[this->n_features * old_t];
        for (int j=0 ; j <  this->n_features; j++) {
            pyx_sum_t[j] = fmax(0, pyx_sum_t[j] - pyx_x[j]);
        }
        // update the py_t array
        double* py_t_t = &py_t[this->n_features * old_t];
        double inv_pt_t = 1.0 / pt[old_t];
        for (int j=0 ; j <  this->n_features; j++) {
            py_t_t[j] = pyx_sum_t[j] * inv_pt_t;
        }

        // ----------- step 2 -  calculate the merge costs and find new_t     ---------

        // get the part of KL1 that relies only on py_x
        double py_x_kl1 = this->py_x_kl[x];

        // loop over the centroids and find the one to which we can add x with the minimal increase in cost
        double min_delta = 0;
        int min_delta_t = -1;
        double old_t_delta = 0;
        const double* py_x_x = &py_x[this->n_features * x];
        for (int t=0 ; t<this->n_clusters ; t++) {
            double p_new = px + pt[t];
            double pi1 = px / p_new;
            double pi2 = 1 - pi1;
            double* py_t_t = &py_t[this->n_features * t];
            double kl1 = py_x_kl1;
            double kl2 = 0;
            for (int j=0 ; j<this->n_features ; j++) {
                double py_t_t_j = py_t_t[j];
                double py_x_x_j = py_x_x[j];
                double average_j = pi1 * py_x_x_j + pi2 * py_t_t_j;
                if (average_j>0) {
                    double log2_inv_average_j = -log2(average_j);
                    kl1 += py_x_x_j * log2_inv_average_j;
                    if (py_t_t_j>0) {
                        kl2 += py_t_t_j * (log2(py_t_t_j) + log2_inv_average_j);
                    }
                }
            }
            double js = pi1 * kl1 + pi2 * kl2;

            if (this->use_inv_beta) {
                double ent = pi1 * log2(pi1) + pi2 * log2(pi2);
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
        for (int j=0 ; j <  this->n_features; j++) {
            pyx_sum_new_t[j] += pyx_x[j];
        }
        // update the py_t array
        double* py_t_new_t = &py_t[this->n_features * new_t];
        double inv_pt_new_t = 1.0 / pt[new_t];
        for (int j=0 ; j <  this->n_features; j++) {
            py_t_new_t[j] = pyx_sum_new_t[j] * inv_pt_new_t;
        }

        // update the pt_x array and changes counter (if need be)
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

double SIBOptimizerDense::calc_labels_costs_score(const double* pt, const double* pyx_sum, double* py_t,
                                                  int new_n_samples, const double* new_py_x, int* labels,
                                                  double* costs, bool infer_mode) {
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

        const double* py_x_x = &new_py_x[this->n_features * x];

        // calculate the part of KL1 that relies only on py_x
        double py_x_kl1 = 0;
        for (int j=0 ; j<this->n_features ; j++) {
            double py_x_x_j = py_x_x[j];
            if (py_x_x_j>0) {
                py_x_kl1 += py_x_x_j * log2(py_x_x_j);
            }
        }

        // loop over the centroids and find the one to which we can add x with the minimal increase in cost
        double min_delta = 0;
        int min_delta_t = -1;
        for (int t=0 ; t<this->n_clusters ; t++) {
            double p_new = px + pt[t];
            double pi1 = px / p_new;
            double pi2 = 1 - pi1;
            const double* py_t_t = &py_t[this->n_features * t];
            double kl1 = py_x_kl1;
            double kl2 = 0;
            for (int j=0 ; j<this->n_features ; j++) {
                double py_t_t_j = py_t_t[j];
                double py_x_x_j = py_x_x[j];
                double average_j = pi1 * py_x_x_j + pi2 * py_t_t_j;
                if (average_j>0) {
                    double log2_inv_average_j = -log2(average_j);
                    kl1 += py_x_x_j * log2_inv_average_j;
                    if (py_t_t_j>0) {
                        kl2 += py_t_t_j * (log2(py_t_t_j) + log2_inv_average_j);
                    }
                }
            }
            double js = pi1 * kl1 + pi2 * kl2;

            if (this->use_inv_beta) {
                double ent = pi1 * log2(pi1) + pi2 * log2(pi2);
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
