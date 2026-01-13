#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define USE_FP64 1
#if USE_FP64
#define real_type double
#else
#define real_type float
#endif

#pragma once

#ifndef V100
#define V100 0
#endif
#ifndef NOACC
#define NOACC 0
#endif
#ifndef CUDA
#define CUDA 1
#endif
#ifndef OPENMP
#define OPENMP 0
#endif
#ifndef HIP
#define HIP 0
#endif

typedef struct {

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
  real_type *d_r; // d_r = 1./d
  int n;

  real_type *aux_vec1;
  real_type *aux_vec2;
  real_type *aux_vec3;

  char *prec_op;
  int m;  // m is outer loop
  int k;  // k is inner loop
} pdata;

void prec_function(int *ia,
                   int *ja,
                   real_type *a,
                   int nnzA,
                   pdata *prec_data,
                   real_type *x,
                   real_type *y);

void cg(int n,
        real_type nnz,
        int *ia,
        int *ja,
        real_type *a,
        real_type *x,
        real_type *b,
        real_type tol,
        pdata *prec_data,
        int maxit,
        int *it,
        int *flag,
        real_type *res_norm_history);

/* preconditioners */

void GS_std(int *ia,
            int *ja,
            real_type *a,
            int nnzA,
            pdata *prec_data,
            real_type *vec_in,
            real_type *vec_out);

void GS_it(int *ia,
           int *ja,
           real_type *a,
           int nnzA,
           pdata *prec_data,
           real_type *vec_in,
           real_type *vec_out);

void GS_it2(int *ia,
            int *ja,
            real_type *a,
            int nnzA,
            pdata *prec_data,
            real_type *vec_in,
            real_type *vec_out);

void it_jacobi(int *ia,
               int *ja,
               real_type *a,
               int nnzA,
               pdata *prec_data,
               real_type *vec_in,
               real_type *vec_out);

void line_jacobi(int *ia,
                 int *ja,
                 real_type *a,
                 int nnzA,
                 pdata *prec_data,
                 real_type *vec_in,
                 real_type *vec_out);
