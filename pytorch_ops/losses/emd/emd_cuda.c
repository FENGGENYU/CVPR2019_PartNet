#include <THC/THC.h>

#include "emd_cuda_kernel.h"

extern THCState *state;

int approxmatch_cuda_forward(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *match, THCudaTensor *temp)
{
    int b = THCudaTensor_size(state, xyz1, 0);
    int b1 = THCudaTensor_size(state, xyz2, 0);
    if (b != b1)
    {
        return 0;
    }
    int n = THCudaTensor_size(state, xyz1, 1);
    int m = THCudaTensor_size(state, xyz2, 1);
    float *xyz1_data = THCudaTensor_data(state, xyz1);
    float *xyz2_data = THCudaTensor_data(state, xyz2);
    float *match_data = THCudaTensor_data(state, match);
    float *temp_data = THCudaTensor_data(state, temp);
    approxmatch_forward_Launcher(b, n, m, xyz1_data, xyz2_data, match_data, temp_data);
    return 1;
}

int matchcost_cuda_forward(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *match, THCudaTensor *out)
{
    int b = THCudaTensor_size(state, xyz1, 0);
    int n = THCudaTensor_size(state, xyz1, 1);
    int m = THCudaTensor_size(state, xyz2, 1);
    float *xyz1_data = THCudaTensor_data(state, xyz1);
    float *xyz2_data = THCudaTensor_data(state, xyz2);
    float *match_data = THCudaTensor_data(state, match);
    float *out_data = THCudaTensor_data(state, out);
    matchcost_forward_Launcher(b, n, m, xyz1_data, xyz2_data, match_data, out_data);
    return 1;
}

int matchcost_cuda_backward(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *match, THCudaTensor *grad1, THCudaTensor *grad2)
{
    int b = THCudaTensor_size(state, xyz1, 0);
    int n = THCudaTensor_size(state, xyz1, 1);
    int m = THCudaTensor_size(state, xyz2, 1);
    float *xyz1_data = THCudaTensor_data(state, xyz1);
    float *xyz2_data = THCudaTensor_data(state, xyz2);
    float *match_data = THCudaTensor_data(state, match);
    float *grad1_data = THCudaTensor_data(state, grad1);
    float *grad2_data = THCudaTensor_data(state, grad2);
    matchcost_backward_Launcher(b, n, m, xyz1_data, xyz2_data, match_data, grad1_data, grad2_data);
    return 1;
}
