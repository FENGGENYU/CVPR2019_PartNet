#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_utils.h"
#include "interpolation_cuda_kernel.h"

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
__global__ void three_interpolate_kernel(int b, int c, int n, int m,
                                        const float *__restrict__ points,
                                        const int *__restrict__ idx,
                                        const float *__restrict__ weight,
                                        float *__restrict__ out) {

    int batch_index = blockIdx.x;
    points += batch_index * m * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    out += batch_index * n * c;

    for (int i = threadIdx.y; i < c; i += blockDim.y) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float w1 = weight[j * 3 + 0];
            float w2 = weight[j * 3 + 1];
            float w3 = weight[j * 3 + 2];

            int i1 = idx[j * 3 + 0];
            int i2 = idx[j * 3 + 1];
            int i3 = idx[j * 3 + 2];

            out[i * blockDim.x + j] = points[i * m + i1] * w1 + points[i * m + i2] * w2 + points[i * m + i3] * w3;
        }
    }
}

// input: grad_out(b, c, n), idxs(b, n, 3), weights(b, n, 3)
// output: grad_points(b, c, m)
__global__ void three_interpolate_grad_kernel(int b, int c, int n, int m,
                                            const float *__restrict__ grad_out,
                                            const int *__restrict__ idx,
                                            const float *__restrict__ weight,
                                            float *__restrict__ grad_points) {
    const int batch_index = blockIdx.x;
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;

    for (int i = threadIdx.y; i < c; i += blockDim.y) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float w1 = weight[j * 3 + 0];
            float w2 = weight[j * 3 + 1];
            float w3 = weight[j * 3 + 2];

            int i1 = idx[j * 3 + 0];
            int i2 = idx[j * 3 + 1];
            int i3 = idx[j * 3 + 2];

            atomicAdd(grad_points + i * m + i1, grad_out[i * n + j] * w1);
            atomicAdd(grad_points + i * m + i2, grad_out[i * n + j] * w2);
            atomicAdd(grad_points + i * m + i3, grad_out[i * n + j] * w3);
        }
    }
}

// input: features(b, c, m), idxs(b, n, 3), weights(b, n, 3)
// output: out(b, c, n)
int three_interpolate_kernel_wrapper(int b, int c, int n, int m, const float *points, const int *idx, const float *weight, float *out) {
    three_interpolate_kernel<<<b, opt_block_config(n, c)>>>(b, c, n, m, points, idx, weight, out);
    return 1;
}

// input: grad_out(b, c, n), idxs(b, n, k), weights(b, n, k)
// output: grad_points(b, c, m)
int three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points) {
    three_interpolate_grad_kernel<<<b, opt_block_config(n, c)>>>(b, c, n, m, grad_out, idx, weight, grad_points);
    return 1;
}

#ifdef __cplusplus
}
#endif
