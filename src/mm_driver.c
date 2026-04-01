#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "common.h"
#include "blas.h"
#include "io_utils.h"

#if CUDA
#include "cuda_blas.h"
#include "devMem.h"
#endif
#if OPENMP
#include "openmp_blas.h"
#endif

#if HIP
#include "hip_blas.h"
#include "devMem.h"
#endif

int main(int argc, char *argv[]) { 
  /* read matrix and optional rhs */

  real_type time_spmv = 0.0;
  struct timeval t1, t2;

  const char *matrixFileName = argv[1];
  int n_trials = atoi(argv[2]);

  mmatrix *A; 
  A = (mmatrix *)calloc(1, sizeof(mmatrix));

  read_mm_file(matrixFileName, A);
  coo_to_csr(A); 

  printf("\n\n");
  printf("Matrix info: \n");
  printf("\n");
  printf("\t Matrix size       : %d x %d \n", A->n, A->n);
  printf("\t Matrix nnz        : %d  \n", A->nnz);
  printf("\t Matrix nnz un     : %d  \n", A->nnz_unpacked);
  printf("\t Number of trials  : %d  \n", n_trials);
#if 1  
  real_type *x = (real_type *) calloc (A->n, sizeof(real_type));
  /* y is RESULT */
  real_type *y = (real_type *) calloc (A->n, sizeof(real_type));
  for (int i = 0; i < A->n; ++i) {
    x[i] = 1.0;   
    y[i] = 0.0;   
  }  
  real_type one = 1.0;
  real_type zero = 0.0;
#if (CUDA || HIP)
  initialize_handles();
  real_type *d_x, *d_y;
  d_x = (real_type*) mallocForDevice (d_x, A->n, sizeof(real_type));
  d_y = (real_type*) mallocForDevice (d_y, A->n, sizeof(real_type));
  memcpyDevice(d_x, x, A->n, sizeof(real_type), "H2D");
  memcpyDevice(d_y, y, A->n, sizeof(real_type), "H2D");

  free(x);
  free(y);
  x = d_x;
  y = d_y;
  int *d_A_ia;
  int *d_A_ja;
  real_type * d_A_a;

  d_A_ia = (int *)  mallocForDevice ((d_A_ia), (A->n + 1), sizeof(int));
  d_A_ja = (int *)  mallocForDevice ((d_A_ja), (A->nnz_unpacked), sizeof(int));
  d_A_a = (real_type *)  mallocForDevice ((d_A_a), (A->nnz_unpacked), sizeof(real_type));
  memcpyDevice(d_A_ia, A->csr_ia, sizeof(int), (A->n + 1), "H2D");
  memcpyDevice(d_A_ja, A->csr_ja , sizeof(int) , (A->nnz_unpacked), "H2D");
  memcpyDevice(d_A_a, A->csr_vals , sizeof(real_type) , (A->nnz_unpacked), "H2D");

  free(A->csr_ia);
  free(A->csr_ja);
  free(A->csr_vals);
  A->csr_ia = d_A_ia;
  A->csr_ja = d_A_ja;
  A->csr_vals = d_A_a;
#if CUDA 
//  printf("initializin spmv buffer \n"); 
  initialize_spmv_buffer(A->n, 
                         A->nnz_unpacked,
                         A->csr_ia,
                         A->csr_ja,
                         A->csr_vals,
                         x,
                         y, 
                         &one, 
                         &zero);
#else // HIP
  analyze_spmv(A->n, 
               A->nnz_unpacked, 
               A->csr_ia,
               A->csr_ja,
               A->csr_vals,
               x,
               y, 
               "A");
#endif
#endif

  gettimeofday(&t1, 0);
  for (int i = 0; i < n_trials; ++i) {
    csr_matvec(A->n, 
               A->nnz_unpacked,   
               A->csr_ia, 
               A->csr_ja, 
               A->csr_vals, 
               x, 
               y, 
               &one, 
               &zero, 
               "A");
  }
  gettimeofday(&t2, 0);
  time_spmv = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

  real_type time_spmv_seconds = time_spmv / 1000.0;
  printf("\n\n");
  printf("SpMV test summary results: \n");
  printf("\n");
  printf("\t Time (total)        : %2.16f  \n", time_spmv_seconds);
  printf("\t Time (av., per one) : %2.16f  \n", time_spmv_seconds / (real_type) n_trials);
  printf("\t Norm of A*x         : %16.16e  \n", sqrt(dot(A->n, y, y)));
  printf("\n\n");
#endif  
  return 0;
}
