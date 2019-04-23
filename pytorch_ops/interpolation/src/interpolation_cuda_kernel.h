#ifdef __cplusplus
extern "C" {
#endif

// input: features(b, c, m), idxs(b, n, 3), weights(b, n, 3)
// output: out(b, c, n)
int three_interpolate_kernel_wrapper(int b, int c, int n, int m, const float *points, const int *idx, const float *weight, float *out);
// input: grad_out(b, c, n), idxs(b, n, k), weights(b, n, k)
// output: grad_points(b, c, m)
int three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points);

#ifdef __cplusplus
}
#endif