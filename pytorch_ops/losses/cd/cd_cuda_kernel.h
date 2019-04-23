#ifndef _CD_KERNEL
#define _CD_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int cd_forward_Launcher(int b, int n, const float *xyz, int m, const float *xyz2, float *result, int *result_i, float *result2, int *result2_i);
int cd_backward_Launcher(int b, int n, const float *xyz1, int m, const float *xyz2, const float *grad_dist1, const int *idx1, const float *grad_dist2, const int *idx2, float *grad_xyz1, float *grad_xyz2);

#ifdef __cplusplus
}
#endif

#endif
