#include <THC/THC.h>
#include "interpolation_cuda_kernel.h"

extern THCState *state;

int three_interpolate_wrapper(int b, int c, int n, int m,
                            THCudaTensor *points_tensor,
                            THCudaIntTensor *idx_tensor,
                            THCudaTensor *weight_tensor,
                            THCudaTensor *out_tensor) {

    const float *points = THCudaTensor_data(state, points_tensor);
    const float *weight = THCudaTensor_data(state, weight_tensor);
    float *out = THCudaTensor_data(state, out_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);

    three_interpolate_kernel_wrapper(b, c, n, m, points, idx, weight, out);
    return 1;
}

int three_interpolate_grad_wrapper(int b, int c, int n, int m,
                                THCudaTensor *grad_out_tensor,
                                THCudaIntTensor *idx_tensor,
                                THCudaTensor *weight_tensor,
                                THCudaTensor *grad_points_tensor) {

    const float *grad_out = THCudaTensor_data(state, grad_out_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);
    const float *weight = THCudaTensor_data(state, weight_tensor);
    float *grad_points = THCudaTensor_data(state, grad_points_tensor);

    three_interpolate_grad_kernel_wrapper(b, c, n, m, grad_out, idx, weight, grad_points);
    return 1;
}