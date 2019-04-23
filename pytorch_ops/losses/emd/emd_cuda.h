int approxmatch_cuda_forward(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *match, THCudaTensor *temp);
int matchcost_cuda_forward(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *match, THCudaTensor *out);
int matchcost_cuda_backward(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *match, THCudaTensor *grad1, THCudaTensor *grad2);