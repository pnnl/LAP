#pragma once

real_type dot(const int n,
              const real_type *v,
              const real_type *w);

void axpy(const int n,
          const real_type alpha,
          real_type *x,
          real_type *y);

void scal(const int n,
          const real_type alpha,
          real_type *v);

void csr_matvec(const int n,
                const int nnz,
                const int *ia,
                const int *ja,
                const real_type *a,
                const real_type *x,
                real_type *result,
                const real_type *al,
                const real_type *bet,
                const char *kind);

void lower_triangular_solve(const int n,
                            const int nnz,
                            const int *lia,
                            const int *lja,
                            const real_type *la,
                            const real_type *diag,
                            const real_type *x,
                            real_type *result);

void upper_triangular_solve(const int n,
                            const int nnz,
                            const int *uia,
                            const int *uja,
                            const real_type *ua,
                            const real_type *diag,
                            const real_type *x,
                            real_type *result);


void gemv(const char *T,
          const int m,
          const int n,
          const double *alpha,
          const double *A,
          const int lda,
          const double *x,
          const double *beta,
          double *y);

void ichol(const int *ia,
           const int *ja,
           real_type *a,
           const int nnzA,
           pdata *prec_data,
           real_type *x,
           real_type *y);

void vec_vec(const int n,
             const real_type *x,
             real_type *y,
             real_type *res);

void vector_reciprocal(const int n,
                       const real_type *v,
                       real_type *res);

void vector_sqrt(const int n,
                 const real_type *v,
                 real_type *res);

void vec_copy(const int n,
              real_type *src,
              real_type *dest);

void vec_zero(const int n,
              real_type *vec);

/* GEMM: C = alpha * op(A) * op(B) + beta * C */
void gemm(const char *transA,
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

/* SYGV: Generalized symmetric eigenvalue problem */
/* Solves A*x = lambda*B*x where A and B are symmetric, B is positive definite */
/* On exit: A contains eigenvectors, w contains eigenvalues (ascending order) */
/* work is workspace of size lwork, iwork is integer workspace of size liwork */
void dsygv(const int n,
           real_type *A,      /* On entry: symmetric matrix A, on exit: eigenvectors */
           real_type *B,      /* On entry: SPD matrix B, destroyed on exit */
           real_type *w,      /* Output: eigenvalues (n) */
           real_type *eigvecs /* Output: eigenvectors (n x n) column-major */
          );

/* SYEV: Standard symmetric eigenvalue problem */
/* Solves A*x = lambda*x where A is symmetric */
void dsyev(const int n,
           real_type *A,      /* On entry: symmetric matrix A, destroyed on exit */
           real_type *w,      /* Output: eigenvalues (n) */
           real_type *eigvecs /* Output: eigenvectors (n x n) column-major */
          );

/* Vector norm (2-norm) */
real_type nrm2(const int n, const real_type *v);

/* Set all elements of vector to a value */
void vec_set(const int n, real_type value, real_type *vec);
