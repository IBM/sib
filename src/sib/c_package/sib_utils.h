/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SIB_UTILS_H_
#define SIB_UTILS_H_


#include <random>
#include <string>


namespace Utils {
    double entropy(const double *a, int size);
    double entropy_safe(const double *a, int size);
    void normalize(int n_samples, int* csr_indptr, int* csr_indices, double* csr_data, int csr_data_size, double* csr_data_out);
    void shuffle(int* array, int size, std::mt19937& random_generator);
    void dump(std::string file_name, const double* array, int size);
    void dump(std::string file_name, const int* array, int size);
};

#endif /* SIB_UTILS_H_ */
