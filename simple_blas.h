//prototypes
//
#include "common.h"
#ifndef SIMPLEBLAS_H
#define SIMPLEBLAS_H
real_type simple_dot (const int n, const real_type *v, const real_type *w);

void simple_scal (const int n, const real_type alpha, real_type *v);

void simple_axpy (const int n, const real_type alpha, const real_type *x, real_type *y);

void simple_csr_matvec(const int n, 
                       const int nnz, 
                       const int *ia, 
                       const int *ja, 
                       const real_type *a, 
                       const real_type *x, 
                       real_type *result, 
                       const  real_type *al, 
                       const real_type *bet);

void simple_lower_triangular_solve(const int n, 
                                   const int nnz,
                                   const int *lia,
                                   const int *lja, 
                                   const real_type *la,
                                   const real_type *diag, 
                                   const real_type *x, 
                                   real_type *result);

void simple_upper_triangular_solve(const int n,
                                   const int nnz, 
                                   const int *uia, 
                                   const int *uja, 
                                   const real_type *ua, 
                                   const real_type *diag,
                                   const real_type *x, 
                                   real_type *result);

void simple_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res);

void simple_vector_sqrt(const int n, const real_type *v, real_type *res);

void simple_vector_reciprocal(const int n, const real_type *v, real_type *res);

void simple_vec_copy(const int n, const real_type *src, real_type *dest);

void simple_vec_zero(const int n, real_type *vec);

void simple_gemv(const char *T,
                 const int m,
                 const int n,
                 const double *alpha,
                 const double *A,
                 const int lda,
                 const double *x,
                 const double *beta,
                 double *y);

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a, 
                      int *lia,
                      int *lja,
                      real_type *la);

void simple_ichol(const int *ia, const int *ja, real_type *a, int nnzA, pdata *prec_data, real_type *x, real_type *y);

/* GEMM: C = alpha * op(A) * op(B) + beta * C */
void simple_gemm(const char *transA,
                 const char *transB,
                 const int m,
                 const int n,
                 const int k,
                 const real_type *alpha,
                 const real_type *A,
                 const int lda,
                 const real_type *B,
                 const int ldb,
                 const real_type *beta,
                 real_type *C,
                 const int ldc);

/* Generalized symmetric eigenvalue problem: A*x = lambda*B*x */
void simple_dsygv(const int n,
                  real_type *A,
                  real_type *B,
                  real_type *w,
                  real_type *eigvecs);

/* Standard symmetric eigenvalue problem: A*x = lambda*x */
void simple_dsyev(const int n,
                  real_type *A,
                  real_type *w,
                  real_type *eigvecs);

/* Vector 2-norm */
real_type simple_nrm2(const int n, const real_type *v);

/* Set vector elements to value */
void simple_vec_set(const int n, real_type value, real_type *vec);

#endif
