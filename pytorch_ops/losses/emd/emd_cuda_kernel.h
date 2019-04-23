#ifdef __cplusplus
extern "C" {
#endif

int approxmatch_forward_Launcher(int b, int n, int m, const float *xyz1, const float *xyz2, float *match, float *temp);
int matchcost_forward_Launcher(int b, int n, int m, const float *xyz1, const float *xyz2, const float *match, float *out);
int matchcost_backward_Launcher(int b, int n, int m, const float *xyz1, const float *xyz2, const float *match, float *grad1, float *grad2);

#ifdef __cplusplus
}
#endif

