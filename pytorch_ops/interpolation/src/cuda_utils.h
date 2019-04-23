#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cmath>
#define TOTAL_THREADS 512

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads = max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

// #define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// #define CHECK_INPUT_TYPE(x, y) AT_ASSERT(x.type().scalarType() == y, #x " must be " #y)

#endif
/*

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <algorithm>
#include <complex>
#define TOTAL_THREADS 512

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads = std::max(std::min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

#endif
*/
