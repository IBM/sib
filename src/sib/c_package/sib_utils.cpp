/*
 * © Copyright IBM Corporation 2020.
 *
 * LICENSE: Apache License 2.0 (Apache-2.0)
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "sib_utils.h"

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

double Utils::entropy(const double *a, int size) {
    double result = 0.0;
    double a_i = 0.0;
    for (int i=0 ; i<size ; i++) {
        a_i = a[i];
        result += a_i * log2(a_i);
    }
    return -result;
}

double Utils::entropy_safe(const double *a, int size) {
    double result = 0.0;
    double a_i = 0.0;
    for (int i=0 ; i<size ; i++) {
        a_i = a[i];
        if (a_i > 0) {
            result += a_i * log2(a_i);
        }
    }
    return -result;
}

void Utils::normalize(int n_samples, int* csr_indptr, int* csr_indices, double* csr_data, int csr_data_size, double* csr_data_out) {
    for (int i=0 ; i<n_samples ; i++) {
        // extract the starting and ending position of the current vector
        int start_indptr = csr_indptr[i];
        int end_indptr = csr_indptr[i + 1];

        // the sum of the vector counts
        double sum = 0;
        for (int j=start_indptr ; j<end_indptr; j++) {
            sum += csr_data[j];
        }

        // divide each entry by the sum
        double inv_sum = 1.0 / sum;
        for (int j=start_indptr ; j<end_indptr; j++) {
            csr_data_out[j] = csr_data[j] * inv_sum;
        }
    }
}

void Utils::shuffle(int* array, int size, std::mt19937& random_generator) {
    std::uniform_int_distribution<int> dis(0,size-1);
    for (int i=0 ; i<size ; i++) {
        int randomPosition = dis(random_generator);
        int temp = array[i];
        array[i] = array[randomPosition];
        array[randomPosition] = temp;
    }
}


void Utils::dump(std::string file_name, const double* array, int size) {
    std::ofstream fout(file_name, std::ios::out);
    fout<<std::fixed << std::setprecision(8);
    for (int i=0 ; i<size ; i++) {
        fout<<array[i]<<std::endl;
    }
    fout.close();
}

void Utils::dump(std::string file_name, const int* array, int size) {
    std::ofstream fout(file_name, std::ios::out);
    fout<<std::fixed << std::setprecision(8);
    for (int i=0 ; i<size ; i++) {
        fout<<array[i]<<std::endl;
    }
    fout.close();
}
