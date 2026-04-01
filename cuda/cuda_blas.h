
#include "common.h"
#ifndef CUDABLAS_H
#define CUDABLAS_H

#ifdef __cplusplus
extern "C" {
#endif

real_type cuda_dot (const int n, const real_type *v, const real_type *w);

void cuda_scal (const int n, const real_type alpha, real_type *v);

void cuda_axpy (const int n, const real_type alpha, const real_type *x, real_type *y);

void cuda_csr_matvec(const int n, 
                     const int nnz, 
                     const int *ia, 
                     const int *ja, 
                     const real_type *a, 
                     const real_type *x, 
                     real_type *result, 
                     const  real_type *al, 
                     const real_type *bet);

void cuda_lower_triangular_solve(const int n,
                                 const int nnz,
                                 const int *lia,
                                 const int *lja,
                                 const real_type *la,
                                 const real_type *diag, 
                                 const real_type *x,
                                 real_type *result);

void cuda_upper_triangular_solve(const int n, 
                                 const int nnz, 
                                 const int *uia, 
                                 const int *uja, 
                                 const real_type *ua, 
                                 const real_type *diag,
                                 const real_type *x, 
                                 real_type *result);

void cuda_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res);

void cuda_vector_sqrt(const int n, const real_type *v, real_type *res);

void cuda_vector_reciprocal(const int n, const real_type *v, real_type *res);

void cuda_vec_copy(const int n, const real_type *src, real_type *dest);

void cuda_vec_zero(const int n, real_type *vec);

void cuda_gemv(const char *T,
               const int m,
               const int n,
               const double *alpha,
               const double *A,
               const int lda,
               const double *x,
               const double *beta,
               double *y);

void initialize_handles();
void finalize_handles();

void initialize_spmv_buffer(const int n, 
                            const int nnz, 
                            int *ia, 
                            int *ja, 
                            real_type *a, 
                            const real_type *x, 
                            real_type *result, 
                            real_type *al, 
                            real_type *bet);


void initialize_and_analyze_L_and_U_solve(const int n, 
                                          const int nnzL, 
                                          int *lia, 
                                          int *lja, 
                                          real_type *la,
                                          const int nnzU, 
                                          int *uia, 
                                          int *uja, 
                                          real_type *ua);

void initialize_L_and_U_descriptors(const int n, 
                                  const int nnzL, 
                                  int *lia, 
                                  int *lja, 
                                  real_type *la,
                                  const int nnzU, 
                                  int *uia, 
                                  int *uja, 
                                  real_type *ua);

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a);

void cuda_ichol(const int *ia, 
                const int *ja, 
                real_type *a, 
                const int nnzA, 
                pdata *prec_data, 
                real_type *x, 
                real_type *y);

/* GEMM: C = alpha * op(A) * op(B) + beta * C */
void cuda_gemm(const char *transA,
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
void cuda_dsygv(const int n,
                real_type *A,
                real_type *B,
                real_type *w,
                real_type *eigvecs);

/* Standard symmetric eigenvalue problem: A*x = lambda*x */
void cuda_dsyev(const int n,
                real_type *A,
                real_type *w,
                real_type *eigvecs);

/* Vector 2-norm */
real_type cuda_nrm2(const int n, const real_type *v);

/* Set vector elements to value */
void cuda_vec_set(const int n, real_type value, real_type *vec);

/* Generate random vectors on GPU */
void cuda_generate_random_vectors(real_type *d_vec, int64_t n, int nev, unsigned long long seed);

#ifdef __cplusplus
}
#endif

#endif
