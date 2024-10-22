#include <cstring>
#include "devMem.h" 
#include <cstdio>
#include "common.h"
#if HIP
#include <hip/hip_runtime_api.h>
#endif

void* mallocForDevice(void *x, int n, int size){
#if CUDA
  cudaError  t = cudaMalloc ((void **)&x,n * size);
#elif  HIP
  hipError_t  t = hipMalloc ((void **) &x,n * size);
#endif
  return x;
}

void memcpyDevice (void *dest, void *src, int n, int size, char *type){
  if (strcmp("H2D", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyHostToDevice);
#elif HIP
    hipMemcpy(dest, src, size * n, hipMemcpyHostToDevice);
#endif
  }
  if (strcmp("D2H", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
#elif HIP
    hipMemcpy(dest, src, size * n, hipMemcpyDeviceToHost);
#endif
  }
  if (strcmp("D2D", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
#elif HIP
    hipMemcpy(dest, src, size * n, hipMemcpyDeviceToHost);
#endif
  }
}

void freeDevice(void *p){
#if CUDA
  cudaFree(p);
#elif HIP
  hipFree(p);
#endif
}
