#ifdef __cplusplus
extern "C" {
#endif

int queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
int selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out);
int groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
int groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points);

#ifdef __cplusplus
}
#endif
