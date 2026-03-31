#ifndef LOBPCG_H
#define LOBPCG_H

#include "common.h"

/* Main LOBPCG function */
void lobpcg(int n, 
            int nnz,
            int *ia,           /* Matrix CSR row pointers */
            int *ja,           /* Matrix CSR column indices */
            real_type *a,      /* Matrix CSR values */
            int nev,           /* Number of eigenvalues requested */
            real_type *X,      /* Initial guess / output eigenvectors (n x nev) */
            real_type *lambda, /* Output eigenvalues (nev) */
            real_type tol,     /* Convergence tolerance */
            pdata *prec_data,  /* Preconditioner data */
            int maxit,         /* Maximum iterations */
            int *it,           /* Output: iterations performed */
            int *nconv,        /* Output: number of converged eigenpairs */
            real_type *res_history, /* Optional: residual history (maxit x nev) */
            int verbose        /* Verbose output: 0=final only, 1=per-iteration details */
           );

/* CGS2 workspace structure for memory reuse */
typedef struct {
  int n_max;          /* Maximum n dimension */
  int k_max;          /* Maximum k dimension */
  real_type *a1;      /* Buffer for projection coefficients (k_max) - host for GPU, main for CPU */
  real_type *a2;      /* Buffer for projection coefficients (k_max) - host for GPU, main for CPU */
  real_type *d_a1;    /* Device buffer for projection coefficients (k_max) - GPU only, NULL for CPU */
  real_type *d_a2;    /* Device buffer for projection coefficients (k_max) - GPU only, NULL for CPU */
} cgs2_workspace;

/* Allocate CGS2 workspace for given maximum dimensions */
cgs2_workspace* cgs2_workspace_alloc(int n_max, int k_max);

/* Free CGS2 workspace */
void cgs2_workspace_free(cgs2_workspace *ws);

/* CGS2 with pre-allocated workspace (for efficiency in loops) */
void cgs2_with_workspace(int n, int k, real_type *V, cgs2_workspace *ws);

/* Compute column norms of a matrix */
void compute_col_norms(int n, int k, const real_type *V, real_type *norms);

/* Sparse matrix - dense matrix multiply: C = alpha * A * B + beta * C */
/* A is sparse (n x n), B is dense (n x k), C is dense (n x k) */
void csr_matmat(int n, int k, int nnz,
                const int *ia, const int *ja, const real_type *a,
                const real_type *B, real_type *C,
                real_type alpha, real_type beta);

#endif /* LOBPCG_H */
