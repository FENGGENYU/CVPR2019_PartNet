int farthestpointsampling_forward_cuda(int b, int n, int m, THCudaTensor *inp, THCudaTensor *temp, THCudaIntTensor *out);
int gatherpoint_forward_cuda(int b, int n, int m, THCudaTensor *inp, THCudaIntTensor *idx, THCudaTensor *out);
int gatherpoint_backward_cuda(int b, int n, int m, THCudaTensor *out_g, THCudaIntTensor *idx, THCudaTensor *inp_g);