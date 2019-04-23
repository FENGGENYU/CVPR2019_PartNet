int three_interpolate_wrapper(int b, int c, int n, int m,
                   THCudaTensor *points_tensor,
                   THCudaIntTensor *idx_tensor,
                   THCudaTensor *weight_tensor,
                   THCudaTensor *out_tensor);

int three_interpolate_grad_wrapper(int b, int c, int n, int m,
                    THCudaTensor *grad_out_tensor,
                    THCudaIntTensor *idx_tensor,
                    THCudaTensor *weight_tensor,
                    THCudaTensor *grad_points_tensor);