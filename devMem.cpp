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
  cudaMemset(x, 0, n * size);
  cudaDeviceSynchronize();
#elif  HIP
  hipError_t  t = hipMalloc ((void **) &x,n * size);
  hipMemset(x, 0, n * size);
  hipDeviceSynchronize();
#endif
  return x;
}

void memcpyDevice (void *dest, void *src, int n, int size, char *type){
  if (strcmp("H2D", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyHostToDevice);
    /* No sync needed for H2D - async execution */
#elif HIP
    hipMemcpy(dest, src, size * n, hipMemcpyHostToDevice);
    /* No sync needed for H2D - async execution */
#endif
  }
  if (strcmp("D2H", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();  /* Sync needed - reading data back to host */
#elif HIP
    hipMemcpy(dest, src, size * n, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();  /* Sync needed - reading data back to host */
#endif
  }
  if (strcmp("D2D", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, size * n, cudaMemcpyDeviceToDevice);
    /* No sync needed for D2D - async execution */
#elif HIP
    hipMemcpy(dest, src, size * n, hipMemcpyDeviceToDevice);
    /* No sync needed for D2D - async execution */
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
