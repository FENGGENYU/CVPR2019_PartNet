#ifdef __cplusplus
extern "C" {
#endif

int farthestpointsamplingLauncher(int b, int n, int m, const float *inp, float *temp, int *out);
int gatherpoint_forward_Launcher(int b, int n, int m, const float *inp, const int *idx, float *out);
int gatherpoint_backward_Launcher(int b, int n, int m, const float *out_g, const int *idx, float *inp_g);

#ifdef __cplusplus
}
#endif
