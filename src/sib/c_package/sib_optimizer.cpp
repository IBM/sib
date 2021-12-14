/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_optimizer.h"
#include <cmath>


// disable the warnings about possible loss of data when converting T to double
#ifdef _MSC_VER
    #pragma warning( disable : 4244 )
#endif

// forward declaration
double fast_log2(double x);


// Constructor
template <typename T>
SIBOptimizer<T>::SIBOptimizer(int32_t n_clusters, int32_t n_features, bool fast_log)
   : n_clusters(n_clusters), n_features(n_features) {
    if (fast_log) {
        this->log2_ptr = fast_log2;
    } else {
        this->log2_ptr = log2;
    }
}

// Destructor
template <typename T>
SIBOptimizer<T>::~SIBOptimizer() {}

template <typename T>
void SIBOptimizer<T>::init_centroids(
        int32_t n_samples, const int32_t *xy_indices,
        const int32_t *xy_indptr, const T *xy_data,
        const T* x_sum, int32_t *labels, bool *x_ignore,
        int32_t *t_size, T *t_sum, double *t_log_sum, T *t_centroid) {

    int32_t x_start = 0;
    int32_t x_end = n_features;
    int32_t x_size = x_end - x_start;
    const int32_t* x_indices;
    const T* x_data;

    bool sparse = xy_indices != NULL;

    for (int32_t x=0; x<n_samples ; x++) {
        if (x_ignore[x]) {
            continue;
        }
        int32_t t = labels[x];
        t_size[t]++;
        t_sum[t] += x_sum[x];

        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &(xy_indices[x_start]);
            x_data = &(xy_data[x_start]);
        } else {
            x_data = &(xy_data[x * n_features]);
        }

        // update t_centroid
        T *t_centroid_t = &t_centroid[t * n_features];
        if (sparse) {
            for (int32_t j=0 ; j<x_size ; j++) {
                t_centroid_t[x_indices[j]] += x_data[j];
            }
        } else {
            for (int32_t j=0 ; j<x_size ; j++) {
                t_centroid_t[j] += x_data[j];
            }
        }
    }

    // set t_log_sum
    for (int32_t t=0; t<n_clusters ; t++) {
        t_log_sum[t] = log2_ptr(t_sum[t]);
    }

}


// sIB iteration over n samples for clustering / classification.
template <typename T>
void SIBOptimizer<T>::iterate(bool clustering_mode,      // clustering / classification mode
        int32_t n_samples, const int32_t *xy_indices,       // data to cluster / classify
        const int32_t *xy_indptr, const T *xy_data,
        const T xy_sum, const T *x_sum,
        int32_t* x_permutation,                             // order of iteration
        int32_t *t_size, T *t_sum,                          // current clusters
        double *t_log_sum, T *t_centroid,
        int32_t *labels, bool *x_locked_in,
        double* costs, double* total_cost, // assigned labels and costs
        double* ity, double* ht, double* change_rate) {     // stats on updates

    int32_t n_changes = 0;

    if (!clustering_mode) {
        *total_cost = 0;
    }

    int32_t x_start = 0;
    int32_t x_end = n_features;
    int32_t x_size = x_end - x_start;
    const int32_t* x_indices;
    const T* x_data;

    bool sparse = xy_indices != NULL;

    for (int32_t i=0; i<n_samples ; i++) {
        int32_t x = clustering_mode ? x_permutation[i] : i;
        if (x_locked_in[x]) {
            continue;
        }
        int32_t old_t = labels[x];

        if (clustering_mode && t_size[old_t] == 1) {
            // skip elements from singleton clusters
            continue;
        }

        // obtain local pointers

        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &(xy_indices[x_start]);
            x_data = &(xy_data[x_start]);
        } else {
            x_data = &(xy_data[x * n_features]);
        }


        T x_sum_x = x_sum[x];

        if (clustering_mode) {
            // withdraw x from its current cluster
            t_size[old_t]--;
            t_sum[old_t] -= x_sum_x;
            t_log_sum[old_t] = log2_ptr(t_sum[old_t]);
            T *old_t_centroid = &(t_centroid[n_features * old_t]);
            if (sparse) {
                for (int32_t j=0 ; j<x_size ; j++) {
                    old_t_centroid[x_indices[j]] -= x_data[j];
                }
            } else {
                for (int32_t j=0 ; j<x_size ; j++) {
                    old_t_centroid[j] -= x_data[j];
                }
            }
        }

        // pointer to the costs array (used only for classification)
        double* x_costs = clustering_mode ? NULL : &costs[this->n_clusters * x];

        double min_cost = 0;
        int32_t min_cost_t = -1;
        double cost_old_t = 0;

        for (int32_t t=0 ; t<this->n_clusters ; t++) {
            T *t_centroid_t = &(t_centroid[n_features * t]);
            T t_sum_t = t_sum[t];
            double log_x_sum_plus_t_sum = log2_ptr(x_sum_x+t_sum_t);
            double t_log_sum_t = t_log_sum[t];
            double sum1 = 0;
            double sum2 = 0;
            double cost = 0;
            if (sparse) {
                for (int32_t j=0 ; j<x_size ; j++) {
                    T t_centroid_t_j = t_centroid_t[x_indices[j]];
                    T x_data_j = x_data[j];
                    T t_centroid_plus_x_j = x_data_j + t_centroid_t_j;
                    double log_t_centroid_plus_x_j = log2_ptr(t_centroid_plus_x_j);
                    sum1 += t_centroid_plus_x_j * (log_x_sum_plus_t_sum - log_t_centroid_plus_x_j);
                    if (t_centroid_t_j > 0) {
                        double log_t_centroid_t_j = log2_ptr(t_centroid_t_j);
                        sum2 += t_centroid_t_j*(log_t_centroid_t_j-log_x_sum_plus_t_sum);
                    }
                }
                cost = sum1 + sum2 + t_sum_t*(log_x_sum_plus_t_sum-t_log_sum_t);
            } else {
                for (int32_t j=0 ; j<x_size ; j++) {
                    T t_centroid_t_j = t_centroid_t[j];
                    T x_data_j = x_data[j];
                    T t_centroid_plus_x_j = x_data_j + t_centroid_t_j;
                    if (t_centroid_plus_x_j > 0) {
                        double log_t_centroid_plus_x_j = log2_ptr(t_centroid_plus_x_j);
                        if (t_centroid_t_j > 0) {
                            double log_t_centroid_t_j = log2_ptr(t_centroid_t_j);
                            sum1 += t_centroid_t_j*(log_t_centroid_t_j-t_log_sum_t);
                        }
                        sum2 += t_centroid_plus_x_j * (log_t_centroid_plus_x_j - log_x_sum_plus_t_sum);
                    }
                }
                cost = sum1 - sum2;
            }
            cost /= xy_sum;

            if (min_cost_t == -1 || cost < min_cost) {
                min_cost_t = t;
                min_cost = cost;
            }

            if (clustering_mode) {
                if (t == old_t) {
                    cost_old_t = cost;
                }
            } else {
                x_costs[t] = cost;
            }
        }

        int32_t new_t = min_cost_t;

        if (clustering_mode) {

            // add x to its new cluster
            t_size[new_t]++;
            t_sum[new_t] += x_sum_x;
            t_log_sum[new_t] = log2_ptr(t_sum[new_t]);
            T *new_t_centroid = &(t_centroid[n_features * new_t]);
            if (sparse) {
                for (int32_t j=0 ; j<x_size ; j++) {
                    new_t_centroid[x_indices[j]] += x_data[j];
                }
            } else {
                for (int32_t j=0 ; j<x_size ; j++) {
                    new_t_centroid[j] += x_data[j];
                }
            }

            if (new_t != old_t) {
                // update the changes counter
                n_changes++;

                // count the increase in information
                *ity += cost_old_t - min_cost;
            }

        } else {
            *total_cost += min_cost;
        }

        labels[x] = new_t;
    }

    if (clustering_mode) {
        // calculate the change rate
        *change_rate = n_samples > 0 ? n_changes / (double)n_samples : 0;

        // calculate the entropy of the clustering analysis
        double ht_sum = 0.0;
        double log_xy_sum = log2_ptr(xy_sum);
        for (int t=0 ; t<this->n_clusters ; t++) {
            T t_sum_t = t_sum[t];
            ht_sum += t_sum_t * (log2_ptr(t_sum_t) - log_xy_sum);
        }
        *ht = -ht_sum / (double)xy_sum;
    }

}

/*
   an approximated log2 implementation, copied from:
   https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-i-the-basics/
   https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-ii-rounding-error/
   https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-iii-the-formulas/

   The idea is to reduce the range to [0.75, 1.5) and to use a rational function, such as
   (ax^2+bx+c)/(dx+e), to approximate log2 on this range.

   We use the code included in the article, which is meant for line 8 in the table therein,
   with a small change - replacing the "if/else" branch by an array.

  Approximation of log2_ptr(float x)

  General idea:
  a) Assuming IEEE representation, x's bit structure is:
     sgn(1):exp(8):frac(23)
     x = (-1)^sgn * (1 + frac) * 2^(exp - 127)
     where 1 + frac is called the significand and is in [1.0, 2.0).

     The frac can be thought of as a fixed-point representation of 23 bits. it ranges
     between 0 (all bits are 0) and 2^23-1 (all bits are 1). So the translation to a float
     significand obtained by 1 + frac / 2^23. When frac = 0, we get 1 + 0 = 1. And when
     frac = 2^23-1, we get 1 + (2^23 - 1)/2^23 = 1 + 1 - 1/2^23, which is close to 2.
     This shows why the significand range is [1, 2).

  b) log2_ptr(x) = logs((-1)^sgn * (1 + frac) * 2^(exp - 127))
             = logs((-1)^sgn) + log2_ptr(1 + frac) +  log2_ptr(2^(exp - 127))
             = log2_ptr(1 + frac) + exp - 127

     Therefore, our job is to calculate log2_ptr(1 + frac) for values in [1, 2).

  c) We follow the set of articles listed above and use an optimal rational function.
     The articles already provide rational functions that were found optimal for significands
     in the range [0.75, 1.5) and we will use the function from line 8 in the table.

  d) Scaling the range of significands from [1.0, 2.0) to [0.75, 1.5) is done by finding a
     new representation for x, in the format x = (-1)^sgn * significand_new * 2^(exp_new-127),
     in which significand_new is in [0.75, 1.5).

     We do this as following:

     case 1) if x's significand is already in [1.0, 1.5], we do not need to do anything and we
     just set significand_new = significand, exp_new = exp.

     case 2) if x's significand is in (1.5, 2.0), we will scale it to [0.75, 1.0) by:
     x = (-1)^sgn * significand * 2^(exp-127)
       = (-1)^sgn * significand * 2^(exp-127) * 2/2
       = (-1)^sgn * (significand/2) * 2^((exp+1)-127)
       = (-1)^sgn * (significand_new) * 2^(exp_new-127)

     So in this case we set significand_new = significand/2, and exp_new = exp + 1

     We distinguish between the case based on the 23rd bit in frac. If it is '1', then the
     value of the significand is greater than 1.5 (case 2). Otherwise, we are in case 1.

  e) The rational function that approximates log2, expects the significand_new and exp_new to
     be passed as:
     significand_new: float offset by -1.0,
     exp_new: float offset by -127.0

  f) To represent significand_new, we create a new number in which the frac bits are taken
     from x, and the exponent bits are set to either 127 (case 1), or 126 (case 2). By
     re-interpreting this as a float, we get for case 1: significand_new = significand, and
     for case 2: significand_new = significand/2.
     We also reduce '1.0' from significand_new as this helps with rounding errors (see the
     original articles).

  g) To represent exp_new, we reduce either 127 (case 1) from exp, or 126 (case 2). In case 2,
     This compensates for dividing the significand by 2.

 */

inline double fast_log2(double x) {
    const float a = 0.338953f;
    const float b = 2.198599f;
    const float c = 1.523692f;
    const unsigned int or_term[2] = {0x3f800000, 0x3f000000};
    const int sub_term[2] = {127, 126};

    float new_significand, new_exponent;
    int exponent;
    union { float f; unsigned int i; } ux1, ux2;
    int is_greater;

    // represent x is a union of unsigned int / float for easier bit analysis
    ux1.f = (float)x;

    // determine if the significand is greater than 1.5 or not - as explained in step d above.
    is_greater = ux1.i & 0x00400000 >> 22;

    // set the new significand - step f above
    ux2.i = (ux1.i & 0x007FFFFF) | or_term[is_greater];
    new_significand = ux2.f - 1.0f;

    // get x's exponent value
    exponent = (ux1.i & 0x7F800000) >> 23;

    // set the new exponent - step g above
    new_exponent = (float)(exponent - sub_term[is_greater]);

    // apply the rational function
    return new_exponent + new_significand * (a*new_significand + b) / (new_significand + c);
}

