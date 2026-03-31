/* Written by KS, Mar 2022 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
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

#if NOACC
#include "simple_blas.h"
#endif
#define MAXIT 50000

int main(int argc, char *argv[])
{
  real_type time_CG = 0.0;
  struct timeval t1, t2;
  srand(12345);

  const char *matrixFileName = argv[1];
  const char *precName = argv[2];

  real_type cg_tol = atof(argv[3]);
  int cg_maxit = atoi(argv[4]);
  int M = atoi(argv[5]);
  int K = atoi(argv[6]);

  printf("argc = %d \n", argc);  
  mmatrix *A, *L, *U, *D;
  A = (mmatrix *)calloc(1, sizeof(mmatrix));
  L = (mmatrix *)calloc(1, sizeof(mmatrix));
  U = (mmatrix *)calloc(1, sizeof(mmatrix));
  D = (mmatrix *)calloc(1, sizeof(mmatrix));

  read_mm_file(matrixFileName, A);
  coo_to_csr(A); 

  split(A, L, U, D);

  /* vector of diagonal elements */

  real_type *d = (real_type *) calloc (A->n, sizeof(real_type));
  for (int i = 0; i < A->n; ++i) {
    d[i] = D->csr_vals[i];   
  }  

  pdata * prec_data;  
  //for(int i=0; i<A->n; ++i) printf("b[%d] = %f\n", i, b[i]);
  prec_data = (pdata*) calloc(1, sizeof(pdata));
  prec_data->n = A->n;
  prec_data->prec_op = (char *) precName;
  prec_data->k = K;
  prec_data->m = M;

  mmatrix * d_A = (mmatrix *)calloc(1, sizeof(mmatrix));
  mmatrix * d_L = (mmatrix *)calloc(1, sizeof(mmatrix));
  mmatrix * d_U = (mmatrix *)calloc(1, sizeof(mmatrix));

  d_A->n = A->n;
  d_A->nnz = A->nnz;
  d_A->nnz_unpacked = A->nnz_unpacked;

  /* now, if the preconditioner is GS_it or GS_it2, the setup is correct but if it is  
   * GS_std, we need to have the diagonal ADDED TO L AND U */
  printf("\n\n");
  printf("Solving linear system for %s\n", matrixFileName);
  if (cg_maxit > MAXIT) {
    printf("\t [ WARNING:]  maxit cannot be larger than %d, re-setting to MAX \n", MAXIT);
    cg_maxit = MAXIT;
  }
  printf("\t Matrix size    : %d x %d \n", A->n, A->n);
  printf("\t Matrix nnz     : %d  \n", A->nnz);
  printf("\t Matrix nnz un  : %d  \n", A->nnz_unpacked);
  printf("\t Preconditioner : %s\n", prec_data->prec_op);
  printf("\t CG tolerance   : %2.16g\n", cg_tol);
  printf("\t CG maxit       : %d \n", cg_maxit);
  printf("\t M (outer it)   : %d \n", M);
  printf("\t K (inner it)   : %d \n", K);
#if (CUDA || HIP)
  if (strcmp(prec_data->prec_op, "GS_std")  == 0) {
    int *new_L_ja = (int *) calloc (L->nnz+L->n, sizeof(int));
    int *new_U_ja = (int *) calloc (U->nnz+U->n, sizeof(int));
    real_type *new_L_a = (real_type *) calloc (L->nnz + L->n, sizeof(real_type));
    real_type *new_U_a = (real_type *) calloc (U->nnz + U->n, sizeof(real_type));

    int c = 0;
    for (int ii = 0; ii < L->n; ++ii) {
      for (int jj = L->csr_ia[ii]; jj < L->csr_ia[ii + 1]; ++jj) {
        new_L_ja[c] = L->csr_ja[jj];
        new_L_a[c] = L->csr_vals[jj];
        c++;     
      }
      /* diagonal element */
      new_L_ja[c] = ii;
      new_L_a[c] = D->csr_vals[ii]; 
      c++;  
    }
    c = 0;
    for (int ii=0; ii<U->n; ++ii){
      /* diagonal element */
      new_U_ja[c] = ii;
      new_U_a[c] = D->csr_vals[ii]; 
      c++;  
      for (int jj = U->csr_ia[ii]; jj < U->csr_ia[ii + 1]; ++jj) {
        new_U_ja[c] = U->csr_ja[jj];
        new_U_a[c] = U->csr_vals[jj];
        c++;     
      }
    }
    /* now shift row pointers */
    for (int ii = 1; ii <= A->n; ++ii) {
      L->csr_ia[ii] += ii;
      U->csr_ia[ii] += ii;
    }
    L->nnz += A->n;
    U->nnz += A->n;

    free(L->csr_ja);
    free(L->csr_vals);

    free(U->csr_ja);
    free(U->csr_vals);
    L->csr_ja = new_L_ja;
    L->csr_vals = new_L_a;
    U->csr_ja = new_U_ja;
    U->csr_vals = new_U_a;
  }/* if */
#endif 
  /* same for ichol on the cpu */
#if NOACC || OPENMP
  if (strcmp(prec_data->prec_op, "ichol")  == 0) {
    printf("READJUSTING L and U \n");
    int *new_L_ja = (int *) calloc (L->nnz + L->n, sizeof(int));
    int *new_U_ja = (int *) calloc (U->nnz + U->n, sizeof(int));
    real_type *new_L_a = (real_type *) calloc (L->nnz + L->n, sizeof(real_type));
    real_type *new_U_a = (real_type *) calloc (U->nnz + U->n, sizeof(real_type));

    int c = 0;
    for (int ii = 0; ii<L->n; ++ii) {
      for (int jj = L->csr_ia[ii]; jj < L->csr_ia[ii + 1]; ++jj) {
        new_L_ja[c] = L->csr_ja[jj];
        new_L_a[c] = L->csr_vals[jj];
        c++;     
      }
      /* diagonal element */
      new_L_ja[c] = ii;
      new_L_a[c] = D->csr_vals[ii]; 
      c++;  
    }
    c = 0;
    for (int ii = 0; ii < U->n; ++ii) {
      /* diagonal element */
      new_U_ja[c] = ii;
      new_U_a[c] = D->csr_vals[ii]; 
      c++;  
      for (int jj = U->csr_ia[ii]; jj < U->csr_ia[ii+1]; ++jj){
        new_U_ja[c] = U->csr_ja[jj];
        new_U_a[c] = U->csr_vals[jj];
        c++;     
      }
    }
    //now shift row pointers
    for (int ii = 1; ii <= A->n; ++ii) {
      L->csr_ia[ii] += ii;
      U->csr_ia[ii] += ii;
    }
    L->nnz += A->n;
    U->nnz += A->n;

    free(L->csr_ja);
    free(L->csr_vals);

    free(U->csr_ja);
    free(U->csr_vals);
    L->csr_ja = new_L_ja;
    L->csr_vals = new_L_a;
    U->csr_ja = new_U_ja;
    U->csr_vals = new_U_a;
  } /* if */
#endif



  real_type *b = (real_type *) calloc (A->n, sizeof(real_type));
  if (argc >7) {//optional rhs file is given
    const char * rhsFileName = argv[7];
    read_rhs(rhsFileName, b);
  } else {
    for (int i = 0; i < A->n; ++i) {
      b[i] = 1.0;
    }
  }

#if (CUDA || HIP)
  initialize_handles();
  real_type *d_b;
  d_b = (real_type*) mallocForDevice (d_b, A->n, sizeof(real_type));
  memcpyDevice(d_b, b, A->n, sizeof(real_type), "H2D");
  free(b);
  b = d_b;
#endif

  prec_data->lnnz = L->nnz;
  prec_data->unnz = U->nnz;

#if (CUDA || HIP)
  prec_data->lia = (int*) mallocForDevice (prec_data->lia,(A->n + 1), sizeof(int));
  prec_data->lja = (int*)  mallocForDevice (prec_data->lja,(L->nnz), sizeof(int));
  prec_data->la = (real_type*) mallocForDevice (prec_data->la,(L->nnz), sizeof(real_type));

  prec_data->uia = (int*)  mallocForDevice (prec_data->uia,(A->n + 1), sizeof(int));
  prec_data->uja = (int*)  mallocForDevice (prec_data->uja,(U->nnz), sizeof(int));
  prec_data->ua = (real_type*)  mallocForDevice (prec_data->ua,(U->nnz), sizeof(real_type));

  memcpyDevice(prec_data->lia,L->csr_ia , (A->n + 1),sizeof(int), "H2D");
  memcpyDevice(prec_data->lja,L->csr_ja , (L->nnz),sizeof(int), "H2D");
  memcpyDevice(prec_data->la,L->csr_vals , (L->nnz),sizeof(real_type), "H2D");

  memcpyDevice(prec_data->uia,U->csr_ia ,(A->n + 1), sizeof(int), "H2D");
  memcpyDevice(prec_data->uja,U->csr_ja , (U->nnz), sizeof(int), "H2D");
  memcpyDevice(prec_data->ua,U->csr_vals , (U->nnz), sizeof(real_type), "H2D");

  prec_data->d_r = (real_type*) mallocForDevice (prec_data->d_r,(A->n), sizeof(real_type));

  //* create dd out of d */
  real_type *d_d;
  d_d = (real_type*) mallocForDevice (d_d, A->n, sizeof(real_type));
  memcpyDevice(d_d, d, A->n,sizeof(real_type), "H2D");
  vector_reciprocal(A->n, d_d, prec_data->d_r);
  free(d);

  prec_data->d = d;

  prec_data->aux_vec1 = (real_type*) mallocForDevice (prec_data->aux_vec1, (A->n), sizeof(real_type));
  prec_data->aux_vec2 = (real_type*) mallocForDevice (prec_data->aux_vec2, (A->n), sizeof(real_type));
  prec_data->aux_vec3 = (real_type*) mallocForDevice (prec_data->aux_vec3, (A->n), sizeof(real_type));

  real_type *x;
  x = (real_type*)  mallocForDevice (x, (A->n), sizeof(real_type));
  vec_zero(A->n, x);  
  int *d_A_ia;
  int *d_A_ja;
  real_type * d_A_a;

  d_A_ia = (int *)  mallocForDevice ((d_A_ia), (A->n+1), sizeof(int));
  d_A_ja = (int *)  mallocForDevice ((d_A_ja), (A->nnz_unpacked), sizeof(int));
  d_A_a = (real_type *)  mallocForDevice ((d_A_a), (A->nnz_unpacked), sizeof(real_type));
  memcpyDevice(d_A_ia, A->csr_ia, sizeof(int), (A->n+1), "H2D");
  memcpyDevice(d_A_ja, A->csr_ja , sizeof(int) , (A->nnz_unpacked), "H2D");
  memcpyDevice(d_A_a, A->csr_vals , sizeof(real_type) , (A->nnz_unpacked), "H2D");
  printf("driver: A->n = %d, A->nnz = %d\n",A->n,  A->nnz_unpacked);
  
  free(A->csr_ia);
  free(A->csr_ja);
  free(A->csr_vals);
  A->csr_ia = d_A_ia;
  A->csr_ja = d_A_ja;
  A->csr_vals = d_A_a;
  real_type one = 1.0; real_type minusone = 1.0;
#if CUDA 
  printf("initializin spmv buffer \n"); 
  initialize_spmv_buffer(A->n, 
                         A->nnz_unpacked,
                         A->csr_ia,
                         A->csr_ja,
                         A->csr_vals,
                         x,
                         b, 
                         &one, 
                         &minusone);
#else // HIP
  analyze_spmv(A->n, 
               A->nnz_unpacked, 
               A->csr_ia,
               A->csr_ja,
               A->csr_vals,
               x,
               b, 
               "A");
  if ((strcmp(prec_data->prec_op, "GS_it")  == 0) || (strcmp(prec_data->prec_op, "GS_it2")  == 0) ) {
    analyze_spmv(A->n, 
                 prec_data->lnnz,
                 prec_data->lia,
                 prec_data->lja,
                 prec_data->la,
                 x,
                 b, 
                 "L");

    analyze_spmv(A->n, 
                 prec_data->unnz,
                 prec_data->uia,
                 prec_data->uja,
                 prec_data->ua,
                 x,
                 b, 
                 "U");
  }
#endif
  if (strcmp(prec_data->prec_op, "GS_std")  == 0) {
    initialize_and_analyze_L_and_U_solve(A->n, 
                                         prec_data->lnnz,
                                         prec_data->lia,
                                         prec_data->lja,
                                         prec_data->la,
                                         prec_data->unnz,
                                         prec_data->uia,
                                         prec_data->uja,
                                         prec_data->ua);
  }

  if (strcmp(prec_data->prec_op, "ichol")  == 0) {
#if (CUDA || HIP)
    prec_data->ichol_vals = (real_type *) mallocForDevice (prec_data->ichol_vals, (A->nnz_unpacked), sizeof(real_type));
    memcpyDevice(prec_data->ichol_vals, A->csr_vals, A->nnz_unpacked,sizeof(real_type), "D2D");

    initialize_ichol(A->n, 
                     A->nnz_unpacked, 
                     A->csr_ia, 
                     A->csr_ja, 
                     prec_data->ichol_vals);
  }
#endif
#if CUDA
  initialize_L_and_U_descriptors(A->n, 
                                 prec_data->lnnz,
                                 prec_data->lia,
                                 prec_data->lja,
                                 prec_data->la,
                                 prec_data->unnz,
                                 prec_data->uia,
                                 prec_data->uja,
                                 prec_data->ua);

#endif
#else //NOT cuda nor hip

  if (strcmp(prec_data->prec_op, "ichol")  == 0) {

    printf("before ICHOL setup \n");
    initialize_ichol(A->n, 
                     U->nnz, 
                     U->csr_ia, 
                     U->csr_ja, 
                     U->csr_vals,
                     L->csr_ia, 
                     L->csr_ja, 
                     L->csr_vals);
  }
  prec_data->lia = L->csr_ia;
  prec_data->lja = L->csr_ja;
  prec_data->la = L->csr_vals;

  prec_data->uia = U->csr_ia;
  prec_data->uja = U->csr_ja;
  prec_data->ua = U->csr_vals;

  real_type *dd = (real_type *) calloc (A->n, sizeof(real_type));
  vector_reciprocal(A->n, d, dd);

  real_type *aux_vec1 = (real_type *) calloc (A->n, sizeof(real_type));
  real_type *aux_vec2 = (real_type *) calloc (A->n, sizeof(real_type));
  real_type *aux_vec3 = (real_type *) calloc (A->n, sizeof(real_type));
  real_type *x = (real_type *) calloc (A->n, sizeof(real_type));
  prec_data->d = d;
  prec_data->d_r = dd;

  prec_data->aux_vec1 = aux_vec1;
  prec_data->aux_vec2 = aux_vec2;
  prec_data->aux_vec3 = aux_vec3;
#endif

  real_type *res_hist = (real_type *) calloc (26000, sizeof(real_type));
  int it, flag;
#if 1
  gettimeofday(&t1, 0);
  cg(A->n,
     A->nnz_unpacked,
     A->csr_ia, //matrix csr data
     A->csr_ja,
     A->csr_vals,
     x, //solution vector, mmust be alocated prior to calling
     b, //rhs
     cg_tol, //DONT MULTIPLY BY NORM OF B
     prec_data, //preconditioner data: all Ls, Us etc
     cg_maxit,
     &it, //output: iteration
     &flag, //output: flag 0-converged, 1-maxit reached, 2-catastrophic failure
     res_hist //output: residual norm history
    );
  gettimeofday(&t2, 0);
  time_CG = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
#endif
  printf("\n\n");
  printf("CG summary results \n");
  printf("\t Iters              : %d  \n", it);
  printf("\t Time               : %2.4f  \n", time_CG/1000.0);
  printf("\t Res. norm          : %2.16g  \n", res_hist[it]);
  printf("\t Preconditioner     : %s\n", prec_data->prec_op);
  if (flag == 0){
    printf("\t Reason for exiting : CG converged  \n");
  } else {
    if (flag == 1){
      printf("\t Reason for exiting : CG reached maxit \n");
    } else {
      printf("\t Reason for exiting : CG failed\n");
    }
  }
  return 0;
}
