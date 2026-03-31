#include <cstring>
#include "devMem.h" 
#include <cstdio>
#include "common.h"
#if HIP
#include <hip/hip_runtime_api.h>
#endif

void* mallocForDevice(void *x, int n, int size){
  if (n <= 0) {
    return NULL;
  }
  /* Cast to size_t to avoid integer overflow for large allocations */
  size_t alloc_size = (size_t)n * (size_t)size;
#if CUDA
  cudaError  t = cudaMalloc ((void **)&x, alloc_size);
  if (t != cudaSuccess) {
    fprintf(stderr, "ERROR: cudaMalloc failed for n=%d, size=%d (%.2f GB)\n", 
            n, size, (double)alloc_size / (1024.0*1024.0*1024.0));
    return NULL;
  }
  cudaMemset(x, 0, alloc_size);
  cudaDeviceSynchronize();
#elif  HIP
  hipError_t  t = hipMalloc ((void **) &x, alloc_size);
  if (t != hipSuccess) {
    fprintf(stderr, "ERROR: hipMalloc failed for n=%d, size=%d (%.2f GB): %s\n", 
            n, size, (double)alloc_size / (1024.0*1024.0*1024.0), hipGetErrorString(t));
    return NULL;
  }
  hipMemset(x, 0, alloc_size);
  hipDeviceSynchronize();
#endif
  return x;
}

void memcpyDevice (void *dest, void *src, int n, int size, char *type){
  /* Cast to size_t to avoid integer overflow for large copies */
  size_t copy_size = (size_t)n * (size_t)size;
  if (strcmp("H2D", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, copy_size, cudaMemcpyHostToDevice);
    /* No sync needed for H2D - async execution */
#elif HIP
    hipMemcpy(dest, src, copy_size, hipMemcpyHostToDevice);
    /* No sync needed for H2D - async execution */
#endif
  }
  if (strcmp("D2H", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, copy_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();  /* Sync needed - reading data back to host */
#elif HIP
    hipMemcpy(dest, src, copy_size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();  /* Sync needed - reading data back to host */
#endif
  }
  if (strcmp("D2D", type) == 0){
#if CUDA
    cudaMemcpy(dest, src, copy_size, cudaMemcpyDeviceToDevice);
    /* No sync needed for D2D - async execution */
#elif HIP
    hipMemcpy(dest, src, copy_size, hipMemcpyDeviceToDevice);
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
