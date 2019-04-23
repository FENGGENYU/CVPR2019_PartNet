#include <THC/THC.h>

#include "sample_cuda_kernel.h"

extern THCState *state;

int farthestpointsampling_forward_cuda(int b, int n, int m, THCudaTensor *inp, THCudaTensor *temp, THCudaIntTensor *out)
{
    float *inp_data = THCudaTensor_data(state, inp);
    float *temp_data = THCudaTensor_data(state, temp);
    int *out_data = THCudaIntTensor_data(state, out);
    farthestpointsamplingLauncher(b, n, m, inp_data, temp_data, out_data);
    return 1;
}

int gatherpoint_forward_cuda(int b, int n, int m, THCudaTensor *inp, THCudaIntTensor *idx, THCudaTensor *out)
{
    float *inp_data = THCudaTensor_data(state, inp);
    int *idx_data = THCudaIntTensor_data(state, idx);
    float *out_data = THCudaTensor_data(state, out);
    gatherpoint_forward_Launcher(b, n, m, inp_data, idx_data, out_data);
    return 1;
}

int gatherpoint_backward_cuda(int b, int n, int m, THCudaTensor *out_g, THCudaIntTensor *idx, THCudaTensor *inp_g)
{
    float *outg_data = THCudaTensor_data(state, out_g);
    int *idx_data = THCudaIntTensor_data(state, idx);
    float *inpg_data = THCudaTensor_data(state, inp_g);
    gatherpoint_backward_Launcher(b, n, m, outg_data, idx_data, inpg_data);
    return 1;
}