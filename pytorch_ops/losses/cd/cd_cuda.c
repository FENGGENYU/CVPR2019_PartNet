#include <THC/THC.h>

#include "cd_cuda_kernel.h"

extern THCState *state;

int cd_forward_cuda(int b, int n, THCudaTensor *xyz, int m, THCudaTensor *xyz2, THCudaTensor *result, THCudaIntTensor *result_i, THCudaTensor *result2, THCudaIntTensor *result2_i)
{
    float *xyz_data = THCudaTensor_data(state, xyz);
    float *xyz2_data = THCudaTensor_data(state, xyz2);
    float *result_data = THCudaTensor_data(state, result);
    int *result_i_data = THCudaIntTensor_data(state, result_i);
    float *result2_data = THCudaTensor_data(state, result2);
    int *result2_i_data = THCudaIntTensor_data(state, result2_i);
    cd_forward_Launcher(b, n, xyz_data, m, xyz2_data, result_data, result_i_data, result2_data, result2_i_data);
    return 1;
}

int cd_backward_cuda(int b, int n, THCudaTensor *xyz1, int m, THCudaTensor *xyz2, THCudaTensor *grad_dist1, THCudaIntTensor *idx1, THCudaTensor *grad_dist2, THCudaIntTensor *idx2, THCudaTensor *grad_xyz1, THCudaTensor *grad_xyz2)
{
    float *xyz1_data = THCudaTensor_data(state, xyz1);
    float *xyz2_data = THCudaTensor_data(state, xyz2);
    float *grad_dist1_data = THCudaTensor_data(state, grad_dist1);
    float *grad_dist2_data = THCudaTensor_data(state, grad_dist2);
    int *idx1_data = THCudaIntTensor_data(state, idx1);
    int *idx2_data = THCudaIntTensor_data(state, idx2);
    float *grad_xyz1_data = THCudaTensor_data(state, grad_xyz1);
    float *grad_xyz2_data = THCudaTensor_data(state, grad_xyz2);
    cd_backward_Launcher(b, n, xyz1_data, m, xyz2_data, grad_dist1_data, idx1_data, grad_dist2_data, idx2_data, grad_xyz1_data, grad_xyz2_data);
    return 1;
}
