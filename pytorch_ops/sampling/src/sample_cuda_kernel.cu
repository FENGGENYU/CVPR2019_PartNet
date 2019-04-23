#ifdef __cplusplus
extern "C" {
#endif

#include "sample_cuda_kernel.h"


  __global__ void farthestpointsamplingKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
    if (m<=0)
      return;
    const int BlockSize=512;
    __shared__ float dists[BlockSize];
    __shared__ int dists_i[BlockSize];
    const int BufferSize=3072;
    __shared__ float buf[BufferSize*3];
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
      int old=0;
      if (threadIdx.x==0)
        idxs[i*m+0]=old;
      for (int j=threadIdx.x;j<n;j+=blockDim.x){
        temp[blockIdx.x*n+j]=1e38;
      }
      for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
        buf[j]=dataset[i*n*3+j];
      }
      __syncthreads();
      for (int j=1;j<m;j++){
        int besti=0;
        float best=-1;
        float x1=dataset[i*n*3+old*3+0];
        float y1=dataset[i*n*3+old*3+1];
        float z1=dataset[i*n*3+old*3+2];
        for (int k=threadIdx.x;k<n;k+=blockDim.x){
          float td=temp[blockIdx.x*n+k];
          float x2,y2,z2;
          if (k<BufferSize){
            x2=buf[k*3+0];
            y2=buf[k*3+1];
            z2=buf[k*3+2];
          }else{
            x2=dataset[i*n*3+k*3+0];
            y2=dataset[i*n*3+k*3+1];
            z2=dataset[i*n*3+k*3+2];
          }
          float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
          float d2=min(d,td);
          if (d2!=td)
            temp[blockIdx.x*n+k]=d2;
          if (d2>best){
            best=d2;
            besti=k;
          }
        }
        dists[threadIdx.x]=best;
        dists_i[threadIdx.x]=besti;
        for (int u=0;(1<<u)<blockDim.x;u++){
          __syncthreads();
          if (threadIdx.x<(blockDim.x>>(u+1))){
            int i1=(threadIdx.x*2)<<u;
            int i2=(threadIdx.x*2+1)<<u;
            if (dists[i1]<dists[i2]){
              dists[i1]=dists[i2];
              dists_i[i1]=dists_i[i2];
            }
          }
        }
        __syncthreads();
        old=dists_i[0];
        if (threadIdx.x==0)
          idxs[i*m+j]=old;
      }
    }
  }
  
  __global__ void gatherpointKernel(int b,int n,int m,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
      for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
        int a=idx[i*m+j];
        out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
        out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
        out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
      }
    }
  }
  
  __global__ void scatteraddpointKernel(int b,int n,int m,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
      for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
        int a=idx[i*m+j];
        atomicAdd(&inp_g[(i*n+a)*3+0],out_g[(i*m+j)*3+0]);
        atomicAdd(&inp_g[(i*n+a)*3+1],out_g[(i*m+j)*3+1]);
        atomicAdd(&inp_g[(i*n+a)*3+2],out_g[(i*m+j)*3+2]);
      }
    }
  }
  
  //require 32*n working space
int farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
    farthestpointsamplingKernel<<<32,512>>>(b,n,m,inp,temp,out);
    return 1;
}

int gatherpoint_forward_Launcher(int b,int n,int m,const float * inp,const int * idx,float * out){
    gatherpointKernel<<<dim3(2,8,1),512>>>(b,n,m,inp,idx,out);
    return 1;
}

int gatherpoint_backward_Launcher(int b,int n,int m,const float * out_g,const int * idx,float * inp_g){
    scatteraddpointKernel<<<dim3(2,8,1),512>>>(b,n,m,out_g,idx,inp_g);
    return 1;
}

#ifdef __cplusplus
}
#endif
