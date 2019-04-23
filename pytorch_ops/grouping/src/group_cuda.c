#include <THC/THC.h>

#include "group_cuda_kernel.h"

extern THCState *state;

int queryBallPoint_cuda(int b, int n, int m, float radius, int nsample, THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaIntTensor *idx, THCudaIntTensor *pts_cnt)
{
    float *xyz1_data = THCudaTensor_data(state, xyz1);
    float *xyz2_data = THCudaTensor_data(state, xyz2);
    int *idx_data = THCudaIntTensor_data(state, idx);
    int *pts_data = THCudaIntTensor_data(state, pts_cnt);
    queryBallPointLauncher(b, n, m, radius, nsample, xyz1_data, xyz2_data, idx_data, pts_data);
    return 1;
}

int selectionSort_cuda(int b, int n, int m, int k, THCudaTensor *dist, THCudaIntTensor *outi, THCudaTensor *out)
{
    float *dist_data = THCudaTensor_data(state, dist);
    int *outi_data = THCudaIntTensor_data(state, outi);
    float *out_data = THCudaTensor_data(state, out);
    selectionSortLauncher(b, n, m, k, dist_data, outi_data, out_data);
    return 1;
}

int groupPoint_forward_cuda(int b, int n, int c, int m, int nsample, THCudaTensor *points, THCudaIntTensor *idx, THCudaTensor *out)
{
    float *points_data = THCudaTensor_data(state, points);
    int *idx_data = THCudaIntTensor_data(state, idx);
    float *out_data = THCudaTensor_data(state, out);
    groupPointLauncher(b, n, c, m, nsample, points_data, idx_data, out_data);
    return 1;
}

int groupPoint_backward_cuda(int b, int n, int c, int m, int nsample, THCudaTensor *grad_out, THCudaIntTensor *idx, THCudaTensor *grad_points)
{
    float *grad_out_data = THCudaTensor_data(state, grad_out);
    int *idx_data = THCudaIntTensor_data(state, idx);
    float *grad_points_data = THCudaTensor_data(state, grad_points);
    groupPointGradLauncher(b, n, c, m, nsample, grad_out_data, idx_data, grad_points_data);
    return 1;
}
