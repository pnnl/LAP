/*
 * LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
 * Eigenvalue solver implementation in C
 * 
 * Based on the simplified GPU-friendly LOBPCG algorithm
 * Supports HIP, CUDA, OpenMP and CPU backends
 */

#include "common.h"
#include "blas.h"
#include "lobpcg.h"
#include <string.h>

#if (CUDA || HIP)
#include "devMem.h"
#endif

/* Small constant for numerical stability */
#define LOBPCG_EPS 1e-8

/*
 * CGS2: Classical Gram-Schmidt with re-orthogonalization
 * Orthonormalizes columns of V in-place
 * V is n x k stored in column-major order
 */
void cgs2(int n, int k, real_type *V) {
  real_type nrm;
  real_type one = 1.0;
  real_type zero = 0.0;
  real_type neg_one = -1.0;
  
  /* Allocate workspace for projection coefficients */
  /* a1 and a2 are vectors of size k (max needed is k-1) */
#if (CUDA || HIP)
  real_type *h_a1 = (real_type *) malloc(k * sizeof(real_type));
  real_type *h_a2 = (real_type *) malloc(k * sizeof(real_type));
  real_type *h_vi = (real_type *) malloc(n * sizeof(real_type));
  real_type *d_a1, *d_a2;
  d_a1 = (real_type *) mallocForDevice(d_a1, k, sizeof(real_type));
  d_a2 = (real_type *) mallocForDevice(d_a2, k, sizeof(real_type));
#else
  real_type *a1 = (real_type *) malloc(k * sizeof(real_type));
  real_type *a2 = (real_type *) malloc(k * sizeof(real_type));
#endif
  
  /* Normalize first column */
  nrm = nrm2(n, V);
  if (nrm > LOBPCG_EPS) {
    scal(n, 1.0 / nrm, V);
  }
  
  /* Process remaining columns using block CGS2 */
  for (int i = 1; i < k; ++i) {
    real_type *vi = V + i * n;  /* Pointer to column i */
    real_type *Vprev = V;       /* V(:,1:i-1) has i columns starting at V */
    
#if (CUDA || HIP)
    /* 
     * CGS2 block orthogonalization:
     * a1 = V(:,1:i-1)' * V(:,i)  -->  GEMV: a1 = Vprev^T * vi
     * V(:,i) = V(:,i) - V(:,1:i-1) * a1  -->  GEMV: vi = vi - Vprev * a1
     * a2 = V(:,1:i-1)' * V(:,i)  -->  reorthogonalize
     * V(:,i) = V(:,i) - V(:,1:i-1) * a2
     */
    
    /* First pass: a1 = Vprev^T * vi */
    /* GEMM: (i x n) * (n x 1) = (i x 1), but we use GEMV equivalent */
    /* gemm("T", "N", i, 1, n, &one, Vprev, n, vi, n, &zero, d_a1, i) */
    gemm("T", "N", i, 1, n, &one, Vprev, n, vi, n, &zero, d_a1, i);
    
    /* vi = vi - Vprev * a1 */
    /* gemm("N", "N", n, 1, i, &neg_one, Vprev, n, d_a1, i, &one, vi, n) */
    gemm("N", "N", n, 1, i, &neg_one, Vprev, n, d_a1, i, &one, vi, n);
    
    /* Second pass (reorthogonalization): a2 = Vprev^T * vi */
    gemm("T", "N", i, 1, n, &one, Vprev, n, vi, n, &zero, d_a2, i);
    
    /* vi = vi - Vprev * a2 */
    gemm("N", "N", n, 1, i, &neg_one, Vprev, n, d_a2, i, &one, vi, n);
    
#else
    /* CPU version using GEMM */
    
    /* First pass: a1 = Vprev^T * vi */
    gemm("T", "N", i, 1, n, &one, Vprev, n, vi, n, &zero, a1, i);
    
    /* vi = vi - Vprev * a1 */
    gemm("N", "N", n, 1, i, &neg_one, Vprev, n, a1, i, &one, vi, n);
    
    /* Second pass (reorthogonalization): a2 = Vprev^T * vi */
    gemm("T", "N", i, 1, n, &one, Vprev, n, vi, n, &zero, a2, i);
    
    /* vi = vi - Vprev * a2 */
    gemm("N", "N", n, 1, i, &neg_one, Vprev, n, a2, i, &one, vi, n);
#endif
    
    /* Normalize column i */
    nrm = nrm2(n, vi);
    if (nrm > LOBPCG_EPS) {
      scal(n, 1.0 / nrm, vi);
    } else {
      /* Vector is nearly zero after orthogonalization - set to small random vector */
      /* This prevents numerical issues with zero columns */
      vec_set(n, 0.0, vi);
      vi[i % n] = 1.0;  /* Set one element to 1 to avoid zero vector */
    }
  }
  
  /* Free workspace */
#if (CUDA || HIP)
  free(h_a1);
  free(h_a2);
  free(h_vi);
  freeDevice(d_a1);
  freeDevice(d_a2);
#else
  free(a1);
  free(a2);
#endif
}

/*
 * Compute column norms of matrix V (n x k)
 */
void compute_col_norms(int n, int k, const real_type *V, real_type *norms) {
  for (int j = 0; j < k; ++j) {
    const real_type *col = V + j * n;
    norms[j] = sqrt(dot(n, col, col));
  }
}

/*
 * Normalize columns of matrix V (n x k)
 */
void normalize_cols(int n, int k, real_type *V) {
  for (int j = 0; j < k; ++j) {
    real_type *col = V + j * n;
    real_type nrm = sqrt(dot(n, col, col));
    if (nrm > LOBPCG_EPS) {
      scal(n, 1.0 / nrm, col);
    }
  }
}

/*
 * Sparse matrix - dense matrix multiply: C = alpha * A * B + beta * C
 * A is sparse CSR (n x n), B is dense column-major (n x k), C is dense column-major (n x k)
 */
void csr_matmat(int n, int k, int nnz,
                const int *ia, const int *ja, const real_type *a,
                const real_type *B, real_type *C,
                real_type alpha, real_type beta) {
  /* Multiply column by column */
  for (int j = 0; j < k; ++j) {
    const real_type *bj = B + j * n;
    real_type *cj = C + j * n;
    real_type al = alpha;
    real_type bt = beta;
    csr_matvec(n, nnz, ia, ja, a, bj, cj, &al, &bt, "A");
  }
}

/*
 * Allocate LOBPCG workspace
 */
lobpcg_data* lobpcg_alloc(int n, int nev, int maxit) {
  lobpcg_data *data = (lobpcg_data*) calloc(1, sizeof(lobpcg_data));
  
  data->n = n;
  data->nev = nev;
  data->maxit = maxit;
  
  int max_subspace = 3 * nev;  /* Maximum subspace size [X, W, P] */
  
#if (CUDA || HIP)
  /* Allocate on device */
  data->X = (real_type*) mallocForDevice(data->X, n * nev, sizeof(real_type));
  data->AX = (real_type*) mallocForDevice(data->AX, n * nev, sizeof(real_type));
  data->W = (real_type*) mallocForDevice(data->W, n * nev, sizeof(real_type));
  data->P = (real_type*) mallocForDevice(data->P, n * nev, sizeof(real_type));
  data->R = (real_type*) mallocForDevice(data->R, n * nev, sizeof(real_type));
  data->X_lock = (real_type*) mallocForDevice(data->X_lock, n * nev, sizeof(real_type));
  data->lambda_lock = (real_type*) mallocForDevice(data->lambda_lock, nev, sizeof(real_type));
  
  /* Small dense matrices - on host for eigenvalue solve */
  data->AS = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  data->BS = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  data->Y = (real_type*) calloc(max_subspace * nev, sizeof(real_type));
  data->theta = (real_type*) calloc(max_subspace, sizeof(real_type));
  
  /* Workspace for dense eigenvalue solve (host) */
  data->lwork = 10 * max_subspace * max_subspace;
  data->work = (real_type*) calloc(data->lwork, sizeof(real_type));
  data->liwork = 10 * max_subspace;
  data->iwork = (int*) calloc(data->liwork, sizeof(int));
  
  /* Residual norms and eigenvalues on host */
  data->res_norms = (real_type*) calloc(nev, sizeof(real_type));
  data->lambda = (real_type*) calloc(nev, sizeof(real_type));
  
  /* Temporary matrices on device */
  data->temp_nn = (real_type*) mallocForDevice(data->temp_nn, n * max_subspace, sizeof(real_type));
  data->temp_kk = (real_type*) mallocForDevice(data->temp_kk, max_subspace * max_subspace, sizeof(real_type));
  
#else
  /* Allocate on host */
  data->X = (real_type*) calloc(n * nev, sizeof(real_type));
  data->AX = (real_type*) calloc(n * nev, sizeof(real_type));
  data->W = (real_type*) calloc(n * nev, sizeof(real_type));
  data->P = (real_type*) calloc(n * nev, sizeof(real_type));
  data->R = (real_type*) calloc(n * nev, sizeof(real_type));
  data->X_lock = (real_type*) calloc(n * nev, sizeof(real_type));
  data->lambda_lock = (real_type*) calloc(nev, sizeof(real_type));
  
  data->AS = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  data->BS = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  data->Y = (real_type*) calloc(max_subspace * nev, sizeof(real_type));
  data->theta = (real_type*) calloc(max_subspace, sizeof(real_type));
  
  data->lwork = 10 * max_subspace * max_subspace;
  data->work = (real_type*) calloc(data->lwork, sizeof(real_type));
  data->liwork = 10 * max_subspace;
  data->iwork = (int*) calloc(data->liwork, sizeof(int));
  
  data->res_norms = (real_type*) calloc(nev, sizeof(real_type));
  data->lambda = (real_type*) calloc(nev, sizeof(real_type));
  
  data->temp_nn = (real_type*) calloc(n * max_subspace, sizeof(real_type));
  data->temp_kk = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
#endif
  
  data->n_locked = 0;
  
  return data;
}

/*
 * Free LOBPCG workspace
 */
void lobpcg_free(lobpcg_data *data) {
  if (data == NULL) return;
  
#if (CUDA || HIP)
  freeDevice(data->X);
  freeDevice(data->AX);
  freeDevice(data->W);
  freeDevice(data->P);
  freeDevice(data->R);
  freeDevice(data->X_lock);
  freeDevice(data->lambda_lock);
  freeDevice(data->temp_nn);
  freeDevice(data->temp_kk);
  
  free(data->AS);
  free(data->BS);
  free(data->Y);
  free(data->theta);
  free(data->work);
  free(data->iwork);
  free(data->res_norms);
  free(data->lambda);
#else
  free(data->X);
  free(data->AX);
  free(data->W);
  free(data->P);
  free(data->R);
  free(data->X_lock);
  free(data->lambda_lock);
  free(data->AS);
  free(data->BS);
  free(data->Y);
  free(data->theta);
  free(data->work);
  free(data->iwork);
  free(data->res_norms);
  free(data->lambda);
  free(data->temp_nn);
  free(data->temp_kk);
#endif
  
  free(data);
}

/*
 * Main LOBPCG function
 */
void lobpcg(int n, 
            int nnz,
            int *ia,
            int *ja,
            real_type *a,
            int nev,
            real_type *X,       /* Initial guess / output eigenvectors (n x nev), column-major */
            real_type *lambda,  /* Output eigenvalues (nev) */
            real_type tol,
            pdata *prec_data,
            int maxit,
            int *it,
            int *nconv,
            real_type *res_history) {
  
  real_type one = 1.0;
  real_type zero = 0.0;
  real_type neg_one = -1.0;
  
  int k = nev;  /* Current number of active eigenvectors */
  int k_active;
  int n_locked = 0;
  int iter = 0;
  int has_P = 0;  /* Flag: do we have P from previous iteration? */
  
  /* Workspace allocation */
  int max_subspace = 3 * nev;
  
  /* Host arrays for small dense operations */
  real_type *h_AS = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  real_type *h_BS = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  real_type *h_Y = (real_type*) calloc(max_subspace * max_subspace, sizeof(real_type));
  real_type *h_theta = (real_type*) calloc(max_subspace, sizeof(real_type));
  real_type *h_Lambda = (real_type*) calloc(nev * nev, sizeof(real_type));
  real_type *h_res_norms = (real_type*) calloc(nev, sizeof(real_type));
  real_type *h_lambda = (real_type*) calloc(nev, sizeof(real_type));
  real_type A_norm = 1.0;  /* Will be estimated from first iteration */
  int *locked = (int*) calloc(nev, sizeof(int));
  
  /* Pre-allocated workspace for orthogonalization coefficient matrices */
  real_type *h_orth_coeff = (real_type*) calloc(nev * nev, sizeof(real_type));
  
#if (CUDA || HIP)
  /* Pre-allocate device workspace for orthogonalization */
  real_type *d_orth_coeff;
  d_orth_coeff = (real_type *) mallocForDevice(d_orth_coeff, nev * nev, sizeof(real_type));
  /* Device workspace */
  real_type *d_AX, *d_W, *d_P, *d_R;
  real_type *d_X_lock;
  real_type *d_S, *d_AS_temp;
  real_type *d_Xnew, *d_Pnew;
  real_type *d_temp;
  
  d_AX = (real_type*) mallocForDevice(d_AX, n * nev, sizeof(real_type));
  d_W = (real_type*) mallocForDevice(d_W, n * nev, sizeof(real_type));
  d_P = (real_type*) mallocForDevice(d_P, n * nev, sizeof(real_type));
  d_R = (real_type*) mallocForDevice(d_R, n * nev, sizeof(real_type));
  d_X_lock = (real_type*) mallocForDevice(d_X_lock, n * nev, sizeof(real_type));
  d_S = (real_type*) mallocForDevice(d_S, n * max_subspace, sizeof(real_type));
  d_AS_temp = (real_type*) mallocForDevice(d_AS_temp, n * max_subspace, sizeof(real_type));
  d_Xnew = (real_type*) mallocForDevice(d_Xnew, n * nev, sizeof(real_type));
  d_Pnew = (real_type*) mallocForDevice(d_Pnew, n * nev, sizeof(real_type));
  d_temp = (real_type*) mallocForDevice(d_temp, n * nev, sizeof(real_type));
  
  /* Zero out P initially */
  vec_zero(n * nev, d_P);
  
  real_type *AX = d_AX;
  real_type *W = d_W;
  real_type *P = d_P;
  real_type *R = d_R;
  real_type *X_lock = d_X_lock;
  real_type *S = d_S;
  real_type *AS_temp = d_AS_temp;
  real_type *Xnew = d_Xnew;
  real_type *Pnew = d_Pnew;
  real_type *temp = d_temp;
  
  /* Copy initial X to device if needed - assume it's already there */
#else
  /* Host workspace */
  real_type *AX = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *W = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *P = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *R = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *X_lock = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *S = (real_type*) calloc(n * max_subspace, sizeof(real_type));
  real_type *AS_temp = (real_type*) calloc(n * max_subspace, sizeof(real_type));
  real_type *Xnew = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *Pnew = (real_type*) calloc(n * nev, sizeof(real_type));
  real_type *temp = (real_type*) calloc(n * nev, sizeof(real_type));
#endif

  /* Initial orthonormalization using CGS2 */
  cgs2(n, k, X);
  
  /* Main LOBPCG iteration */
  for (iter = 1; iter <= maxit; ++iter) {
    k_active = k - n_locked;
    
    if (k_active <= 0) {
      printf("LOBPCG: All eigenpairs converged at iteration %d\n", iter);
      break;
    }
    
    /* ------------------------------ */
    /* Apply operator: AX = A * X     */
    /* ------------------------------ */
    csr_matmat(n, k_active, nnz, ia, ja, a, X, AX, 1.0, 0.0);
    
    /* ------------------------------ */
    /* Rayleigh quotient: Lambda = X'*AX */
    /* ------------------------------ */
    /* This gives us a k_active x k_active dense matrix */
    /* Lambda = X' * AX  (k_active x k_active) */
    /* Using GEMM: Lambda = X^T * AX */
#if (CUDA || HIP)
    /* Compute Lambda = X'*AX on device using rocBLAS GEMM, then copy small result to host */
    {
      real_type one_rq = 1.0, zero_rq = 0.0;
      /* Use d_orth_coeff as device buffer for Lambda (k_active x k_active) */
      gemm("T", "N", k_active, k_active, n, &one_rq, X, n, AX, n, &zero_rq, d_orth_coeff, k_active);
      /* Copy result to host */
      memcpyDevice(h_Lambda, d_orth_coeff, k_active * k_active, sizeof(real_type), "D2H");
    }
    
    /* Diagonal elements are the eigenvalue estimates */
    for (int i = 0; i < k_active; ++i) {
      h_lambda[i] = h_Lambda[i + i * k_active];
    }
#else
    /* CPU version: Lambda = X^T * AX using GEMM */
    real_type one_h = 1.0, zero_h = 0.0;
    gemm("T", "N", k_active, k_active, n, &one_h, X, n, AX, n, &zero_h, h_Lambda, k_active);
    
    /* Diagonal elements are the eigenvalue estimates */
    for (int i = 0; i < k_active; ++i) {
      h_lambda[i] = h_Lambda[i + i * k_active];
    }
#endif
    
    /* ------------------------------ */
    /* Residuals: R = AX - X * Lambda (MATLAB style, using full Lambda matrix) */
    /* ------------------------------ */
    /* R = AX - X * Lambda where Lambda = X' * AX (k_active x k_active) */
    {
      real_type one_res = 1.0, neg_one_res = -1.0;
      /* First copy AX to R */
      for (int i = 0; i < k_active; ++i) {
        vec_copy(n, AX + i * n, R + i * n);
      }
      /* Then R = R - X * Lambda = AX - X * Lambda */
      /* h_Lambda is on host, need to use it for GEMM */
#if (CUDA || HIP)
      /* Copy h_Lambda to device for GEMM */
      memcpyDevice(d_orth_coeff, h_Lambda, k_active * k_active, sizeof(real_type), "H2D");
      gemm("N", "N", n, k_active, k_active, &neg_one_res, X, n, d_orth_coeff, k_active, &one_res, R, n);
#else
      gemm("N", "N", n, k_active, k_active, &neg_one_res, X, n, h_Lambda, k_active, &one_res, R, n);
#endif
    }
    
    /* Compute residual norms (absolute, like MATLAB: res = vecnorm(R)) */
    compute_col_norms(n, k_active, R, h_res_norms);
    
    /* Print iteration info */
    real_type max_res = 0.0;
    for (int i = 0; i < k_active; ++i) {
      if (h_res_norms[i] > max_res) max_res = h_res_norms[i];
    }
    printf("LOBPCG: it %4d  max_res=%.8e  lambda[0]=%.8e\n", iter, max_res, h_lambda[0]);
    
    
    if (res_history != NULL) {
      for (int i = 0; i < k_active; ++i) {
        res_history[(iter - 1) * nev + i] = h_res_norms[i];
      }
    }
    
    /* ------------------------------ */
    /* Lock converged eigenpairs      */
    /* ------------------------------ */
    /* Use absolute residual for convergence (like MATLAB: locked = res < tol) */
    int any_locked = 0;
    for (int i = 0; i < k_active; ++i) {
      locked[i] = (h_res_norms[i] < tol) ? 1 : 0;
      if (locked[i]) {
        any_locked = 1;
      }
    }
    
    /* Move locked eigenpairs and compact active vectors */
    if (any_locked) {
      /* First, copy locked eigenpairs to X_lock and lambda */
      for (int i = 0; i < k_active; ++i) {
        if (locked[i]) {
          /* Copy to locked storage with correct column index */
          vec_copy(n, X + i * n, X_lock + n_locked * n);
          lambda[n_locked] = h_lambda[i];
          printf("LOBPCG:   Eigenpair %d converged at iter %d, lambda = %.8e, res = %.8e\n", 
                 n_locked + 1, iter, h_lambda[i], h_res_norms[i]);
          n_locked++;
        }
      }
      
      /* Check if all converged */
      if (n_locked >= nev) {
        printf("LOBPCG: All %d eigenpairs converged\n", nev);
        break;
      }
      
      /* Compact X, R, and P to remove locked columns (like MATLAB: X = X(:, ~locked)) */
      int write_idx = 0;
      for (int i = 0; i < k_active; ++i) {
        if (!locked[i]) {
          if (write_idx != i) {
            vec_copy(n, X + i * n, X + write_idx * n);
            vec_copy(n, R + i * n, R + write_idx * n);
            if (has_P) {
              vec_copy(n, P + i * n, P + write_idx * n);
            }
            h_lambda[write_idx] = h_lambda[i];
          }
          write_idx++;
        }
      }
      
      /* Update k_active after compaction */
      k_active = write_idx;
      
      if (k_active <= 0) {
        printf("LOBPCG: All eigenpairs converged after compaction\n");
        break;
      }
    }
    
    /* ------------------------------ */
    /* Preconditioned residual: W = M^{-1} * R */
    /* ------------------------------ */
    for (int i = 0; i < k_active; ++i) {
#if HIP
      vec_zero(n, W + i * n);
#endif
      prec_function(ia, ja, a, nnz, prec_data, R + i * n, W + i * n);
    }
    
    
    /* Orthogonalize W against X: W = W - X * (X' * W) */
    {
      real_type one_orth = 1.0, zero_orth = 0.0, neg_one_orth = -1.0;
#if (CUDA || HIP)
      gemm("T", "N", k_active, k_active, n, &one_orth, X, n, W, n, &zero_orth, d_orth_coeff, k_active);
      gemm("N", "N", n, k_active, k_active, &neg_one_orth, X, n, d_orth_coeff, k_active, &one_orth, W, n);
#else
      gemm("T", "N", k_active, k_active, n, &one_orth, X, n, W, n, &zero_orth, h_orth_coeff, k_active);
      gemm("N", "N", n, k_active, k_active, &neg_one_orth, X, n, h_orth_coeff, k_active, &one_orth, W, n);
#endif
    }
    
    /* Orthogonalize W against P (like MATLAB: if ~isempty(P), W = W - P*(P'*W)) */
    if (has_P) {
      real_type one_orth = 1.0, zero_orth = 0.0, neg_one_orth = -1.0;
#if (CUDA || HIP)
      gemm("T", "N", k_active, k_active, n, &one_orth, P, n, W, n, &zero_orth, d_orth_coeff, k_active);
      gemm("N", "N", n, k_active, k_active, &neg_one_orth, P, n, d_orth_coeff, k_active, &one_orth, W, n);
#else
      gemm("T", "N", k_active, k_active, n, &one_orth, P, n, W, n, &zero_orth, h_orth_coeff, k_active);
      gemm("N", "N", n, k_active, k_active, &neg_one_orth, P, n, h_orth_coeff, k_active, &one_orth, W, n);
#endif
    }
    
    
    /* Drop tiny W directions (like MATLAB: idxW = vecnorm(W) > 1e-12; W = W(:, idxW)) */
    int kW = 0;
    for (int i = 0; i < k_active; ++i) {
      real_type w_nrm = nrm2(n, W + i * n);
      if (w_nrm > 1e-12) {
        if (kW != i) {
          vec_copy(n, W + i * n, W + kW * n);
        }
        kW++;
      }
    }
    
    /* Orthonormalize W for numerical stability in the eigenvalue solve */
    /* (MATLAB's eig(AS,BS) handles non-orthonormal bases, but our solver needs conditioning) */
    if (kW > 0) {
      cgs2(n, kW, W);
    }
    
    
    /* Drop tiny P directions (like MATLAB: idxP = vecnorm(P) > 1e-12; P = P(:, idxP)) */
    int kP = 0;
    if (has_P) {
      for (int i = 0; i < k_active; ++i) {
        real_type p_nrm = nrm2(n, P + i * n);
        if (p_nrm > 1e-12) {
          if (kP != i) {
            vec_copy(n, P + i * n, P + kP * n);
          }
          kP++;
        }
      }
    }
    
    /* ------------------------------ */
    /* Build trial subspace S = [X, W, P] */
    /* ------------------------------ */
    int kX = k_active;
    int ssize = kX + kW + kP;
    
    /* Copy X to S */
    for (int i = 0; i < kX; ++i) {
      vec_copy(n, X + i * n, S + i * n);
    }
    /* Copy W to S */
    for (int i = 0; i < kW; ++i) {
      vec_copy(n, W + i * n, S + (kX + i) * n);
    }
    /* Copy P to S */
    for (int i = 0; i < kP; ++i) {
      vec_copy(n, P + i * n, S + (kX + kW + i) * n);
    }
    
    /* Compute AS = A * S */
    csr_matmat(n, ssize, nnz, ia, ja, a, S, AS_temp, 1.0, 0.0);
    
    /* ------------------------------ */
    /* Compute Rayleigh-Ritz matrices: G = S'*AS, B = S'*S */
    /* ------------------------------ */
    for (int i = 0; i < ssize; ++i) {
      for (int j = 0; j < ssize; ++j) {
        h_AS[i + j * ssize] = dot(n, S + i * n, AS_temp + j * n);
        h_BS[i + j * ssize] = dot(n, S + i * n, S + j * n);
      }
    }
    
    /* Symmetrize G and B */
    for (int i = 0; i < ssize; ++i) {
      for (int j = i + 1; j < ssize; ++j) {
        real_type avg_g = 0.5 * (h_AS[i + j * ssize] + h_AS[j + i * ssize]);
        real_type avg_b = 0.5 * (h_BS[i + j * ssize] + h_BS[j + i * ssize]);
        h_AS[i + j * ssize] = avg_g;
        h_AS[j + i * ssize] = avg_g;
        h_BS[i + j * ssize] = avg_b;
        h_BS[j + i * ssize] = avg_b;
      }
    }
    
    
    
    /* ------------------------------ */
    /* Solve generalized eigenvalue problem: G*y = theta*B*y */
    /* ------------------------------ */
    dsygv(ssize, h_AS, h_BS, h_theta, h_Y);
    
    
    /* Extract first k_active Ritz vectors */
    /* Y contains eigenvectors as columns, theta contains eigenvalues (sorted ascending) */
    
    /* Partition Y into Yx, Yw, Yp blocks */
    /* Y is ssize x k_active, partitioned as:
     *   Yx: rows 0 to kX-1
     *   Yw: rows kX to kX+kW-1
     *   Yp: rows kX+kW to ssize-1
     */
    
    /* ------------------------------ */
    /* Update X and P (TRUE LOBPCG)   */
    /* ------------------------------ */
    /* Xnew = X * Yx + W * Yw + P * Yp */
    /* Pnew = W * Yw + P * Yp */
    
    /* Compute Xnew = S * Y[:, 0:k_active] using GEMM */
    /* S is n x ssize on device, Y is ssize x ssize on host */
    /* We need Y[:, 0:k_active] which is ssize x k_active */
    {
      real_type one_g = 1.0, zero_g = 0.0;
      
      /* Extract Y_small (ssize x k_active) for Xnew and Yw_Yp (ssize-kX x k_active) for Pnew */
      real_type *h_Y_small = (real_type*) malloc(ssize * k_active * sizeof(real_type));
      for (int j = 0; j < k_active; ++j) {
        for (int i = 0; i < ssize; ++i) {
          h_Y_small[i + j * ssize] = h_Y[i + j * ssize];
        }
      }
      
#if (CUDA || HIP)
      /* Copy Y_small to device */
      real_type *d_Y_small;
      d_Y_small = (real_type*) mallocForDevice(d_Y_small, ssize * k_active, sizeof(real_type));
      memcpyDevice(d_Y_small, h_Y_small, ssize * k_active, sizeof(real_type), "H2D");
      
      /* Xnew = S * Y_small using device GEMM */
      gemm("N", "N", n, k_active, ssize, &one_g, S, n, d_Y_small, ssize, &zero_g, Xnew, n);
      
      /* Pnew = [W, P] * Y[kX:ssize, 0:k_active] */
      /* This is (W, P) * Y_small[kX:ssize, :] */
      vec_zero(n * k_active, Pnew);
      /* W contribution: W * Y[kX:kX+kW, :] */
      if (kW > 0) {
        real_type *d_Yw = d_Y_small + kX;  /* Offset to Yw block */
        gemm("N", "N", n, k_active, kW, &one_g, W, n, d_Yw, ssize, &zero_g, Pnew, n);
      }
      /* P contribution: P * Y[kX+kW:ssize, :] */
      if (kP > 0) {
        real_type *d_Yp = d_Y_small + (kX + kW);  /* Offset to Yp block */
        gemm("N", "N", n, k_active, kP, &one_g, P, n, d_Yp, ssize, &one_g, Pnew, n);
      }
      
      freeDevice(d_Y_small);
#else
      /* CPU version: Xnew = S * Y_small */
      gemm("N", "N", n, k_active, ssize, &one_g, S, n, h_Y_small, ssize, &zero_g, Xnew, n);
      
      /* Pnew = [W, P] * Y[kX:ssize, :] */
      vec_zero(n * k_active, Pnew);
      if (kW > 0) {
        real_type *h_Yw = h_Y_small + kX;
        gemm("N", "N", n, k_active, kW, &one_g, W, n, h_Yw, ssize, &zero_g, Pnew, n);
      }
      if (kP > 0) {
        real_type *h_Yp = h_Y_small + (kX + kW);
        gemm("N", "N", n, k_active, kP, &one_g, P, n, h_Yp, ssize, &one_g, Pnew, n);
      }
#endif
      
      free(h_Y_small);
    }
    
    /* Stabilize Xnew with full orthonormalization (like MATLAB line 129: [Xnew, ~] = cgs2(Xnew)) */
    cgs2(n, k_active, Xnew);
    
    /* Stabilize Pnew: orthogonalize against Xnew using block GEMM */
    /* Pnew = Pnew - Xnew * (Xnew' * Pnew) */
    {
      real_type one_stab = 1.0, zero_stab = 0.0, neg_one_stab = -1.0;
#if (CUDA || HIP)
      gemm("T", "N", k_active, k_active, n, &one_stab, Xnew, n, Pnew, n, &zero_stab, d_orth_coeff, k_active);
      gemm("N", "N", n, k_active, k_active, &neg_one_stab, Xnew, n, d_orth_coeff, k_active, &one_stab, Pnew, n);
#else
      gemm("T", "N", k_active, k_active, n, &one_stab, Xnew, n, Pnew, n, &zero_stab, h_orth_coeff, k_active);
      gemm("N", "N", n, k_active, k_active, &neg_one_stab, Xnew, n, h_orth_coeff, k_active, &one_stab, Pnew, n);
#endif
    }
    
    /* Orthonormalize Pnew (MATLAB line 140: [Pnew, Rp] = cgs2(Pnew)) */
    cgs2(n, k_active, Pnew);
    
    /* ------------------------------ */
    /* Update for next iteration      */
    /* ------------------------------ */
    /* Copy Xnew to X */
    for (int i = 0; i < k_active; ++i) {
      vec_copy(n, Xnew + i * n, X + i * n);
    }
    /* Copy Pnew to P */
    for (int i = 0; i < k_active; ++i) {
      vec_copy(n, Pnew + i * n, P + i * n);
    }
    
    /* Update eigenvalue estimates */
    for (int i = 0; i < k_active; ++i) {
      h_lambda[i] = h_theta[i];
    }
    
    has_P = 1;  /* We now have P for next iteration */
    
  }  /* End main iteration loop */
  
  /* ------------------------------ */
  /* Finalize results               */
  /* ------------------------------ */
  /* Combine locked + active eigenpairs */
  /* Copy remaining active eigenpairs to output */
  for (int i = 0; i < k - n_locked; ++i) {
    vec_copy(n, X + i * n, X + (n_locked + i) * n);
    lambda[n_locked + i] = h_lambda[i];
  }
  
  /* Copy locked eigenvectors to output */
  for (int i = 0; i < n_locked; ++i) {
    vec_copy(n, X_lock + i * n, X + i * n);
  }
  
  /* Sort eigenvalues and eigenvectors by ascending eigenvalue */
  /* Simple insertion sort for small nev */
  for (int i = 0; i < nev - 1; ++i) {
    int min_idx = i;
    for (int j = i + 1; j < nev; ++j) {
      if (lambda[j] < lambda[min_idx]) {
        min_idx = j;
      }
    }
    if (min_idx != i) {
      /* Swap eigenvalues */
      real_type tmp = lambda[i];
      lambda[i] = lambda[min_idx];
      lambda[min_idx] = tmp;
      
      /* Swap eigenvector columns */
      for (int row = 0; row < n; ++row) {
        tmp = X[i * n + row];
        X[i * n + row] = X[min_idx * n + row];
        X[min_idx * n + row] = tmp;
      }
    }
  }
  
  *it = iter;
  *nconv = n_locked;
  
  /* Cleanup */
#if (CUDA || HIP)
  freeDevice(d_AX);
  freeDevice(d_W);
  freeDevice(d_P);
  freeDevice(d_R);
  freeDevice(d_X_lock);
  freeDevice(d_S);
  freeDevice(d_AS_temp);
  freeDevice(d_Xnew);
  freeDevice(d_Pnew);
  freeDevice(d_temp);
#else
  free(AX);
  free(W);
  free(P);
  free(R);
  free(X_lock);
  free(S);
  free(AS_temp);
  free(Xnew);
  free(Pnew);
  free(temp);
#endif
  
  free(h_AS);
  free(h_BS);
  free(h_Y);
  free(h_theta);
  free(h_Lambda);
  free(h_res_norms);
  free(h_lambda);
  free(locked);
  free(h_orth_coeff);
#if (CUDA || HIP)
  freeDevice(d_orth_coeff);
#endif
}
