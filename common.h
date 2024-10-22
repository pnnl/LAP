#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#define USE_FP64 0
#if USE_FP64
#define real_type double
#else
#define real_type float
#endif

#pragma once

#ifndef V100
#	define V100 0
#endif
#ifndef NOACC
#	define NOACC 0
#endif
#ifndef CUDA
#	define CUDA 1
#endif
#ifndef OPENMP
#	define OPENMP 0
#endif
#ifndef HIP
#define HIP 0
#endif

typedef struct{
 
  int *lia;
  int *lja;
  real_type *la;
  int lnnz; 
 
  int *uia;
  int *uja;
  real_type *ua;
  int unnz;

  real_type *ichol_vals;
  real_type *d;
  real_type *d_r;//d_r = 1./d
  int n;

  real_type *aux_vec1, *aux_vec2, *aux_vec3;

  char *prec_op;
  int m, k;//m is outer loop, k inner
} pdata;

void prec_function(int *ia, int *ja, real_type *a, int nnzA,pdata* prec_data, real_type * x, real_type *y);

void cg(int n, real_type nnz,
        int *ia, //matrix csr data
        int *ja,
        real_type *a,
        real_type *x, //solution vector, mmust be alocated prior to calling
        real_type *b, //rhs
        real_type tol, //DONT MULTIPLY BY NORM OF B
        pdata *prec_data, //preconditioner data: all Ls, Us etc
        int maxit,
        int *it, //output: iteration
        int *flag, //output: flag 0-converged, 1-maxit reached, 2-catastrophic failure
        real_type *res_norm_history //output: residual norm history
       );

/* preconditioners */

void GS_std(int *ia, int *ja, real_type *a, int nnzA,  pdata* prec_data, real_type *vec_in, real_type *vec_out);
void GS_it(int *ia, int *ja, real_type *a, int nnzA,  pdata* prec_data, real_type *vec_in, real_type *vec_out);
void GS_it2(int *ia, int *ja, real_type *a, int nnzA,  pdata* prec_data, real_type *vec_in, real_type *vec_out);
void it_jacobi(int *ia, int *ja, real_type *a, int nnzA,  pdata* prec_data, real_type *vec_in, real_type *vec_out);
void line_jacobi(int *ia, int *ja, real_type *a, int nnzA,  pdata* prec_data, real_type *vec_in, real_type *vec_out);
