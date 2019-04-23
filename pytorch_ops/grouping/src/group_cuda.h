int queryBallPoint_cuda(int b, int n, int m, float radius, int nsample, THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaIntTensor *idx, THCudaIntTensor *pts_cnt);
int selectionSort_cuda(int b, int n, int m, int k, THCudaTensor *dist, THCudaIntTensor *outi, THCudaTensor *out);
int groupPoint_forward_cuda(int b, int n, int c, int m, int nsample, THCudaTensor *points, THCudaIntTensor *idx, THCudaTensor *out);
int groupPoint_backward_cuda(int b, int n, int c, int m, int nsample, THCudaTensor *grad_out, THCudaIntTensor *idx, THCudaTensor *grad_points);
