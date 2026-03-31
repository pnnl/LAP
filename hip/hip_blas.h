#include "common.h"
#ifndef HIPBLAS_H
#define HIPBLAS_H

void initialize_handles();

void analyze_spmv(const int n,
                  const int nnz,
                  int *ia,
                  int *ja,
                  real_type *a,
                  const real_type *x,
                  real_type *result,
                  char *option);

void initialize_and_analyze_L_and_U_solve(const int n,
                                          const int nnzL,
                                          int *lia,
                                          int *lja,
                                          real_type *la,
                                          const int nnzU,
                                          int *uia,
                                          int *uja,
                                          real_type *ua);

real_type hip_dot(const int n,
                  const real_type *v,
                  const real_type *w);

void hip_scal(const int n,
              const real_type alpha,
              real_type *v);

void hip_axpy(const int n,
              const real_type alpha,
              const real_type *x,
              real_type *y);

void hip_csr_matvec(const int n,
                    const int nnz,
                    const int *ia,
                    const int *ja,
                    const real_type *a,
                    const real_type *x,
                    real_type *result,
                    const real_type *al,
                    const real_type *bet,
                    const char *kind);

void hip_lower_triangular_solve(const int n,
                                const int nnzL,
                                const int *lia,
                                const int *lja,
                                const real_type *la,
                                const real_type *diagonal,
                                const real_type *x,
                                real_type *result);

void hip_upper_triangular_solve(const int n,
                                const int nnzU,
                                const int *uia,
                                const int *uja,
                                const real_type *ua,
                                const real_type *diagonal,
                                const real_type *x,
                                real_type *result);

void hip_gemv(const char *T,
              const int m,
              const int n,
              const double *alpha,
              const double *A,
              const int lda,
              const double *x,
              const double *beta,
              double *y);

void hip_vec_vec(const int n,
                 const real_type *x,
                 const real_type *y,
                 real_type *res);

void hip_vector_reciprocal(const int n,
                           const real_type *v,
                           real_type *res);

void hip_vector_sqrt(const int n,
                     const real_type *v,
                     real_type *res);

void hip_vec_copy(const int n,
                  const real_type *src,
                  real_type *dest);

void hip_vec_zero(const int n,
                  real_type *vec);

void hip_vec_set(const int n,
                 real_type value,
                 real_type *vec);

void initialize_ichol(const int n,
                      const int nnzA,
                      int *ia,
                      int *ja,
                      real_type *a);

void hip_ichol(const int *ia,
               const int *ja,
               real_type *a,
               const int nnzA,
               pdata *prec_data,
               real_type *x,
               real_type *y);

/* SpMM: Sparse matrix times dense matrix C = alpha * A * B + beta * C */
void hip_csrmm(const int n, const int k, const int nnz,
               const int *ia, const int *ja, const real_type *a,
               const real_type *B, real_type *C,
               const real_type alpha, const real_type beta,
               const char *kind);

/* GEMM: C = alpha * op(A) * op(B) + beta * C */
void hip_gemm(const char *transA,
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
/* NOTE: For small dense matrices, we copy to host and use LAPACK/rocSOLVER */
void hip_dsygv(const int n,
               real_type *A,
               real_type *B,
               real_type *w,
               real_type *eigvecs);

/* Standard symmetric eigenvalue problem: A*x = lambda*x */
void hip_dsyev(const int n,
               real_type *A,
               real_type *w,
               real_type *eigvecs);

/* Vector 2-norm */
real_type hip_nrm2(const int n, const real_type *v);

/* Set vector elements to value - already declared as hip_vec_set */

/* Generate random vectors on GPU using hiprand */
void hip_generate_random_vectors(real_type *d_vec, int64_t n, int nev, unsigned long long seed);

/* Batched column norms using GEMM - computes norms of k columns of V (n x k) */
void hip_compute_col_norms_batched(int n, int k, const real_type *V, real_type *norms);

/* Batched nrm2 using rocblas strided batched */
void hip_nrm2_batched(int n, int k, const real_type *V, real_type *norms);

/* Cholesky QR orthonormalization - O(1) syncs instead of O(k) */
int hip_cholesky_qr(int n, int k, real_type *V);

/* TSQR orthonormalization using rocSOLVER - single sync */
int hip_tsqr(int n, int k, real_type *V);

/* Device-side CGS2 - single sync at end instead of O(k) syncs */
void hip_cgs2_device(int n, int k, real_type *V, real_type eps);

#endif
