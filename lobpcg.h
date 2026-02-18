#ifndef LOBPCG_H
#define LOBPCG_H

#include "common.h"

/* LOBPCG data structure for holding workspace and parameters */
typedef struct {
  int n;            /* Matrix dimension */
  int nev;          /* Number of eigenvalues/eigenvectors requested */
  int maxit;        /* Maximum iterations */
  real_type tol;    /* Convergence tolerance */
  
  /* Workspace arrays (all n x nev or smaller) */
  real_type *X;       /* Current eigenvector approximations (n x nev) */
  real_type *AX;      /* A*X (n x nev) */
  real_type *W;       /* Preconditioned residual (n x nev) */
  real_type *P;       /* Search directions (n x nev) */
  real_type *R;       /* Residuals (n x nev) */
  
  /* Locked eigenpairs */
  real_type *X_lock;        /* Locked eigenvectors */
  real_type *lambda_lock;   /* Locked eigenvalues */
  int n_locked;             /* Number of locked eigenpairs */
  
  /* Small dense matrices for Rayleigh-Ritz */
  real_type *AS;      /* S'*A*S (at most 3*nev x 3*nev) */
  real_type *BS;      /* S'*S (at most 3*nev x 3*nev) */
  real_type *Y;       /* Ritz vectors (at most 3*nev x nev) */
  real_type *theta;   /* Ritz values (at most 3*nev) */
  
  /* Auxiliary workspace for dense eigenvalue problem */
  real_type *work;    /* General workspace */
  int lwork;          /* Workspace size */
  int *iwork;         /* Integer workspace for LAPACK */
  int liwork;         /* Integer workspace size */
  
  /* Column norms / residual norms */
  real_type *res_norms;  /* Residual norms (nev) */
  real_type *lambda;     /* Current eigenvalue estimates (nev) */
  
  /* Auxiliary dense matrix workspace */
  real_type *temp_nn;    /* Temporary n x n or smaller */
  real_type *temp_kk;    /* Temporary k x k */
  
} lobpcg_data;

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

/* CGS2 orthonormalization: orthonormalize columns of V in-place */
void cgs2(int n, int k, real_type *V);

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

/* Normalize columns of a matrix */
void normalize_cols(int n, int k, real_type *V);

/* Sparse matrix - dense matrix multiply: C = alpha * A * B + beta * C */
/* A is sparse (n x n), B is dense (n x k), C is dense (n x k) */
void csr_matmat(int n, int k, int nnz,
                const int *ia, const int *ja, const real_type *a,
                const real_type *B, real_type *C,
                real_type alpha, real_type beta);

/* Allocate LOBPCG workspace */
lobpcg_data* lobpcg_alloc(int n, int nev, int maxit);

/* Free LOBPCG workspace */
void lobpcg_free(lobpcg_data *data);

#endif /* LOBPCG_H */
