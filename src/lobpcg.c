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

#if HIP
/* HIP-specific functions */
extern void hip_generate_random_vectors(real_type *d_vec, int64_t n, int nev, unsigned long long seed);
#endif

/* Small constant for numerical stability (must match CG_experiments: 1e-14) */
#define LOBPCG_EPS 1e-14

/* Uncomment to enable eigenvalue/eigenvector verification at the end of LOBPCG */
/* #define LOBPCG_VERIFY */

/*
 * Allocate CGS2 workspace for given maximum dimensions
 * This allows memory to be allocated once and reused across multiple cgs2 calls
 */
cgs2_workspace* cgs2_workspace_alloc(int n_max, int k_max) {
  cgs2_workspace *ws = (cgs2_workspace*) calloc(1, sizeof(cgs2_workspace));
  ws->n_max = n_max;
  ws->k_max = k_max;
  
  /* Always allocate host buffers */
  ws->a1 = (real_type*) malloc(k_max * sizeof(real_type));
  ws->a2 = (real_type*) malloc(k_max * sizeof(real_type));
  
#if (CUDA || HIP)
  /* Also allocate device buffers for GPU builds */
  ws->d_a1 = (real_type*) mallocForDevice(ws->d_a1, k_max, sizeof(real_type));
  ws->d_a2 = (real_type*) mallocForDevice(ws->d_a2, k_max, sizeof(real_type));
#else
  ws->d_a1 = NULL;
  ws->d_a2 = NULL;
#endif
  
  return ws;
}

/*
 * Free CGS2 workspace
 */
void cgs2_workspace_free(cgs2_workspace *ws) {
  if (ws == NULL) return;
  
  free(ws->a1);
  free(ws->a2);
  
#if (CUDA || HIP)
  freeDevice(ws->d_a1);
  freeDevice(ws->d_a2);
#endif
  
  free(ws);
}

/*
 * Fast orthonormalization - CGS2 with batched norms is fastest
 * TSQR via rocSOLVER has overhead that doesn't pay off for our k values
 */
void ortho_fast(int n, int k, real_type *V, cgs2_workspace *ws) {
  cgs2_with_workspace(n, k, V, ws);
}

/*
 * CGS2: Classical Gram-Schmidt with re-orthogonalization (with pre-allocated workspace)
 * Orthonormalizes columns of V in-place
 * V is n x k stored in column-major order
 * ws: pre-allocated workspace (must have k_max >= k)
 */
void cgs2_with_workspace(int n, int k, real_type *V, cgs2_workspace *ws) {
  real_type nrm;
  real_type one = 1.0;
  real_type zero = 0.0;
  real_type neg_one = -1.0;
  
#if (CUDA || HIP)
  real_type *d_a1 = ws->d_a1;
  real_type *d_a2 = ws->d_a2;
#else
  real_type *a1 = ws->a1;
  real_type *a2 = ws->a2;
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
     * CGS2 using GEMV (like CG_experiments):
     * a1 = V(:,1:i-1)' * V(:,i)  -->  GEMV: a1 = Vprev^T * vi
     * V(:,i) = V(:,i) - V(:,1:i-1) * a1  -->  GEMV: vi = vi - Vprev * a1
     * a2 = V(:,1:i-1)' * V(:,i)  -->  reorthogonalize
     * V(:,i) = V(:,i) - V(:,1:i-1) * a2
     */
    
    /* First pass: a1 = Vprev^T * vi (GEMV with transpose) */
    gemv("T", n, i, &one, Vprev, n, vi, &zero, d_a1);
    
    /* vi = vi - Vprev * a1 (GEMV without transpose) */
    gemv("N", n, i, &neg_one, Vprev, n, d_a1, &one, vi);
    
    /* Second pass (reorthogonalization): a2 = Vprev^T * vi */
    gemv("T", n, i, &one, Vprev, n, vi, &zero, d_a2);
    
    /* vi = vi - Vprev * a2 */
    gemv("N", n, i, &neg_one, Vprev, n, d_a2, &one, vi);
    
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
      /* Vector is nearly zero after orthogonalization */
      /* Set to a small canonical vector to avoid numerical issues */
#if (CUDA || HIP)
      /* For GPU: use a small buffer and copy */
      real_type *h_canonical = (real_type*) calloc(n, sizeof(real_type));
      h_canonical[i % n] = 1.0;
      memcpyDevice(vi, h_canonical, n, sizeof(real_type), "H2D");
      free(h_canonical);
#else
      vec_set(n, 0.0, vi);
      vi[i % n] = 1.0;
#endif
    }
  }
}

/*
 * Compute column norms of matrix V (n x k)
 * Uses batched GEMM approach on GPU to avoid per-column synchronization
 */
void compute_col_norms(int n, int k, const real_type *V, real_type *norms) {
#if (CUDA || HIP)
  /* Use batched version to avoid k separate sync calls */
  compute_col_norms_batched(n, k, V, norms);
#else
  for (int j = 0; j < k; ++j) {
    const real_type *col = V + j * n;
    norms[j] = sqrt(dot(n, col, col));
  }
#endif
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
            real_type *res_history,
            int verbose) {
  
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
  
  /* Estimate memory requirements */
  size_t mem_per_vec = (size_t)n * sizeof(real_type);
  size_t mem_subspace = (size_t)n * max_subspace * sizeof(real_type);
  size_t mem_total = mem_per_vec * (7 * nev)  /* X, AX, W, P, R, Xnew, Pnew, X_lock */
                   + mem_subspace * 2;        /* S, AS_temp */
  
  printf("LOBPCG: Estimated GPU memory requirement: %.2f GB\n", 
         (double)mem_total / (1024.0 * 1024.0 * 1024.0));
  fflush(stdout);
  
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
  
  /* Check for allocation failures */
  if (d_AX == NULL || d_W == NULL || d_P == NULL || d_R == NULL || 
      d_X_lock == NULL || d_S == NULL || d_AS_temp == NULL ||
      d_Xnew == NULL || d_Pnew == NULL || d_temp == NULL) {
    fprintf(stderr, "ERROR: LOBPCG failed to allocate device memory.\n");
    fprintf(stderr, "       Required: %.2f GB for matrix of size %d with %d eigenvalues.\n",
            (double)((size_t)n * (7 * nev + 2 * max_subspace) * sizeof(real_type)) / (1024.0 * 1024.0 * 1024.0),
            n, nev);
    fprintf(stderr, "       Try reducing the number of eigenvalues (nev) or use a smaller matrix.\n");
    *nconv = 0;
    *it = 0;
    return;
  }
  
  /* Pre-allocate device buffers for Rayleigh-Ritz matrices (GEMM optimization) */
  real_type *d_AS_rr, *d_BS_rr;
  d_AS_rr = (real_type*) mallocForDevice(d_AS_rr, max_subspace * max_subspace, sizeof(real_type));
  d_BS_rr = (real_type*) mallocForDevice(d_BS_rr, max_subspace * max_subspace, sizeof(real_type));
  
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

  /* Pre-allocate CGS2 workspace (reused across all cgs2 calls) */
  cgs2_workspace *cgs2_ws = cgs2_workspace_alloc(n, max_subspace);
  
  /* Pre-allocate Y_small buffers for Rayleigh-Ritz projection (reused each iteration) */
  real_type *h_Y_small = (real_type*) malloc(max_subspace * nev * sizeof(real_type));
#if (CUDA || HIP)
  real_type *d_Y_small;
  d_Y_small = (real_type*) mallocForDevice(d_Y_small, max_subspace * nev, sizeof(real_type));
#endif

  /* Pre-allocate lock coefficient buffers (for orthogonalization against locked vectors) */
  /* Size: n_locked * max_subspace (at most nev * 3*nev) */
  real_type *h_lock_coeff = (real_type*) malloc(nev * max_subspace * sizeof(real_type));
#if (CUDA || HIP)
  real_type *d_lock_coeff;
  d_lock_coeff = (real_type*) mallocForDevice(d_lock_coeff, nev * max_subspace, sizeof(real_type));
#else
  real_type *lock_coeff = h_lock_coeff;  /* Alias for CPU version */
#endif

  /* Initial orthonormalization using fast method (Cholesky QR with CGS2 fallback) */
  ortho_fast(n, k, X, cgs2_ws);
  
  /* Main LOBPCG iteration */
  for (iter = 1; iter <= maxit; ++iter) {
    k_active = k - n_locked;
    
    if (k_active <= 0) {
      if (verbose) {
        printf("========================================\n");
        printf("LOBPCG: All eigenpairs converged at iteration %d\n", iter);
      }
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
    
    if (verbose) {
      printf("it %4d  max residual = %.3e\n", iter, max_res);
      for (int i = 0; i < k_active; ++i) {
        printf("  eigenvalue %d: %.10e  residual: %.3e\n", 
               n_locked + i + 1, h_lambda[i], h_res_norms[i]);
      }
    }
    
    
    if (res_history != NULL) {
      for (int i = 0; i < k_active; ++i) {
        res_history[(iter - 1) * nev + i] = h_res_norms[i];
      }
    }
    
    /* ------------------------------ */
    /* Lock converged eigenpairs      */
    /* ------------------------------ */
    /* Use absolute residual for convergence (like MATLAB: locked = res < tol) */
    /* Also check that eigenvector has unit norm (not collapsed to zero) */
    int any_locked = 0;
    for (int i = 0; i < k_active; ++i) {
      locked[i] = 0;  /* Default: not locked */
      
      /* Check residual is below tolerance */
      if (h_res_norms[i] < tol) {
        /* Verify the eigenvector has approximately unit norm (since we orthonormalize) */
        /* After CGS2, norm should be ~1. If norm << 1, eigenvector has collapsed. */
        real_type x_nrm = nrm2(n, X + i * n);
        
        if (x_nrm > 0.5) {
          /* Valid convergence - eigenvector has reasonable norm */
          locked[i] = 1;
          any_locked = 1;
        } else {
          /* Eigenvector collapsed - don't lock */
          if (verbose) {
            printf("  ** Warning: eigenpair %d has degenerate eigenvector (norm=%.2e), skipping lock\n",
                   n_locked + i + 1, x_nrm);
          }
        }
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
          if (verbose) {
            printf("  ** Eigenpair %d converged at iter %d, lambda = %.10e, residual = %.3e\n", 
                   n_locked + 1, iter, h_lambda[i], h_res_norms[i]);
          }
          n_locked++;
        }
      }
      
      /* Check if all converged */
      if (n_locked >= nev) {
        if (verbose) {
          printf("========================================\n");
          printf("LOBPCG: All %d eigenpairs converged at iteration %d\n", nev, iter);
        }
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
        if (verbose) {
          printf("========================================\n");
          printf("LOBPCG: All eigenpairs converged after compaction\n");
        }
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
    
    /* Orthogonalize W and P against locked vectors (like MATLAB lines 64-67) */
    /* This is critical for stability when eigenpairs are locked */
    /* Uses pre-allocated buffers (d_lock_coeff/lock_coeff) */
    if (n_locked > 0) {
      real_type one_orth = 1.0, zero_orth = 0.0, neg_one_orth = -1.0;
      /* W = W - X_lock * (X_lock' * W) */
#if (CUDA || HIP)
      gemm("T", "N", n_locked, k_active, n, &one_orth, X_lock, n, W, n, &zero_orth, d_lock_coeff, n_locked);
      gemm("N", "N", n, k_active, n_locked, &neg_one_orth, X_lock, n, d_lock_coeff, n_locked, &one_orth, W, n);
      
      /* P = P - X_lock * (X_lock' * P) */
      if (has_P) {
        gemm("T", "N", n_locked, k_active, n, &one_orth, X_lock, n, P, n, &zero_orth, d_lock_coeff, n_locked);
        gemm("N", "N", n, k_active, n_locked, &neg_one_orth, X_lock, n, d_lock_coeff, n_locked, &one_orth, P, n);
      }
#else
      gemm("T", "N", n_locked, k_active, n, &one_orth, X_lock, n, W, n, &zero_orth, lock_coeff, n_locked);
      gemm("N", "N", n, k_active, n_locked, &neg_one_orth, X_lock, n, lock_coeff, n_locked, &one_orth, W, n);
      
      if (has_P) {
        gemm("T", "N", n_locked, k_active, n, &one_orth, X_lock, n, P, n, &zero_orth, lock_coeff, n_locked);
        gemm("N", "N", n, k_active, n_locked, &neg_one_orth, X_lock, n, lock_coeff, n_locked, &one_orth, P, n);
      }
#endif
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
    /* Use batched norm computation to avoid k_active sync calls */
    int kW = 0;
    {
      real_type *w_norms = (real_type*) malloc(k_active * sizeof(real_type));
#if (CUDA || HIP)
      nrm2_batched(n, k_active, W, w_norms);
#else
      for (int i = 0; i < k_active; ++i) {
        w_norms[i] = nrm2(n, W + i * n);
      }
#endif
      for (int i = 0; i < k_active; ++i) {
        if (w_norms[i] > 1e-12) {
          if (kW != i) {
            vec_copy(n, W + i * n, W + kW * n);
          }
          kW++;
        }
      }
      free(w_norms);
    }
    
    /* Orthonormalize W for numerical stability in the eigenvalue solve */
    /* (MATLAB's eig(AS,BS) handles non-orthonormal bases, but our solver needs conditioning) */
    if (kW > 0) {
      ortho_fast(n, kW, W, cgs2_ws);
    }
    
    
    /* Drop tiny P directions (like MATLAB: idxP = vecnorm(P) > 1e-12; P = P(:, idxP)) */
    int kP = 0;
    if (has_P) {
      real_type *p_norms = (real_type*) malloc(k_active * sizeof(real_type));
#if (CUDA || HIP)
      nrm2_batched(n, k_active, P, p_norms);
#else
      for (int i = 0; i < k_active; ++i) {
        p_norms[i] = nrm2(n, P + i * n);
      }
#endif
      for (int i = 0; i < k_active; ++i) {
        if (p_norms[i] > 1e-12) {
          if (kP != i) {
            vec_copy(n, P + i * n, P + kP * n);
          }
          kP++;
        }
      }
      free(p_norms);
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
    /* Always use GEMM for efficiency */
    /* ------------------------------ */
#if (CUDA || HIP)
    {
      real_type one_rr = 1.0, zero_rr = 0.0;
      gemm("T", "N", ssize, ssize, n, &one_rr, S, n, AS_temp, n, &zero_rr, d_AS_rr, ssize);
      gemm("T", "N", ssize, ssize, n, &one_rr, S, n, S, n, &zero_rr, d_BS_rr, ssize);
      memcpyDevice(h_AS, d_AS_rr, ssize * ssize, sizeof(real_type), "D2H");
      memcpyDevice(h_BS, d_BS_rr, ssize * ssize, sizeof(real_type), "D2H");
    }
#else
    {
      real_type one_rr = 1.0, zero_rr = 0.0;
      /* CPU: always use GEMM */
      gemm("T", "N", ssize, ssize, n, &one_rr, S, n, AS_temp, n, &zero_rr, h_AS, ssize);
      gemm("T", "N", ssize, ssize, n, &one_rr, S, n, S, n, &zero_rr, h_BS, ssize);
    }
#endif
    
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
    /* Check condition of B - if ill-conditioned, restart without P */
    /* ------------------------------ */
    real_type min_diag_B = fabs(h_BS[0]);
    real_type max_diag_B = fabs(h_BS[0]);
    for (int i = 1; i < ssize; ++i) {
      real_type d = fabs(h_BS[i + i * ssize]);
      if (d < min_diag_B) min_diag_B = d;
      if (d > max_diag_B) max_diag_B = d;
    }
    real_type cond_B = (min_diag_B > 1e-14) ? max_diag_B / min_diag_B : 1e16;
    
    /* If B is severely ill-conditioned and we have P, retry without P */
    if (cond_B > 1e12 && kP > 0) {
      /* Rebuild S without P */
      kP = 0;
      ssize = kX + kW;
      
      /* Recompute AS and BS without P */
      csr_matmat(n, ssize, nnz, ia, ja, a, S, AS_temp, 1.0, 0.0);
      
#if (CUDA || HIP)
      {
        real_type one_rr = 1.0, zero_rr = 0.0;
        gemm("T", "N", ssize, ssize, n, &one_rr, S, n, AS_temp, n, &zero_rr, d_AS_rr, ssize);
        gemm("T", "N", ssize, ssize, n, &one_rr, S, n, S, n, &zero_rr, d_BS_rr, ssize);
        memcpyDevice(h_AS, d_AS_rr, ssize * ssize, sizeof(real_type), "D2H");
        memcpyDevice(h_BS, d_BS_rr, ssize * ssize, sizeof(real_type), "D2H");
      }
#else
      {
        real_type one_rr = 1.0, zero_rr = 0.0;
        gemm("T", "N", ssize, ssize, n, &one_rr, S, n, AS_temp, n, &zero_rr, h_AS, ssize);
        gemm("T", "N", ssize, ssize, n, &one_rr, S, n, S, n, &zero_rr, h_BS, ssize);
      }
#endif
      
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
      
      has_P = 0;  /* Reset P for next iteration */
    }
    
    /* ------------------------------ */
    /* Solve generalized eigenvalue problem: G*y = theta*B*y */
    /* ------------------------------ */
    dsygv(ssize, h_AS, h_BS, h_theta, h_Y);
    
    /* ------------------------------ */
    /* Sort eigenvalues and eigenvectors by ascending eigenvalue */
    /* rocsolver_dsygv does NOT guarantee sorted output */
    /* ------------------------------ */
    {
      /* Simple selection sort for small ssize (typically <= 15) */
      for (int i = 0; i < ssize - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < ssize; ++j) {
          if (h_theta[j] < h_theta[min_idx]) {
            min_idx = j;
          }
        }
        if (min_idx != i) {
          /* Swap eigenvalues */
          real_type tmp_theta = h_theta[i];
          h_theta[i] = h_theta[min_idx];
          h_theta[min_idx] = tmp_theta;
          
          /* Swap eigenvector columns in h_Y (column i and column min_idx) */
          /* h_Y is stored column-major: column j is h_Y[j*ssize : (j+1)*ssize-1] */
          for (int row = 0; row < ssize; ++row) {
            real_type tmp_y = h_Y[i * ssize + row];
            h_Y[i * ssize + row] = h_Y[min_idx * ssize + row];
            h_Y[min_idx * ssize + row] = tmp_y;
          }
        }
      }
    }
    
    /* ------------------------------ */
    /* Sanity check: if eigenvalues are unreasonable, skip this update */
    /* This detects when the eigenvalue solver produced garbage */
    /* ------------------------------ */
    int skip_update = 0;
    
    /* Check for NaN/Inf in eigenvalues */
    for (int i = 0; i < k_active; ++i) {
      if (h_theta[i] != h_theta[i] || h_theta[i] > 1e30 || h_theta[i] < -1e30) {
        skip_update = 1;
        has_P = 0;
        break;
      }
    }
    
    if (skip_update) {
      /* Don't update X, P - just continue with current approximation */
      /* The residual computation next iteration will provide new search direction */
      continue;
    }
    
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
      /* Using pre-allocated h_Y_small buffer (max size: max_subspace * nev) */
      for (int j = 0; j < k_active; ++j) {
        for (int i = 0; i < ssize; ++i) {
          h_Y_small[i + j * ssize] = h_Y[i + j * ssize];
        }
      }
      
#if (CUDA || HIP)
      /* Copy Y_small to pre-allocated device buffer */
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
    }
    
    /* Stabilize Xnew with full orthonormalization (like MATLAB line 129: [Xnew, ~] = cgs2(Xnew)) */
    ortho_fast(n, k_active, Xnew, cgs2_ws);
    
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
    ortho_fast(n, k_active, Pnew, cgs2_ws);
    
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
  /* For eigenpairs that haven't converged, use current h_lambda estimates */
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
  
  /* ------------------------------ */
  /* Optional eigenvalue/eigenvector verification */
  /* Define LOBPCG_VERIFY to enable this check */
  /* ------------------------------ */
#ifdef LOBPCG_VERIFY
  {
    printf("\n=== LOBPCG Verification ===\n");
    
    /* Allocate temporary vector for A*x */
    real_type *Ax_verify;
#if (CUDA || HIP)
    Ax_verify = (real_type*) mallocForDevice(Ax_verify, n, sizeof(real_type));
#else
    Ax_verify = (real_type*) malloc(n * sizeof(real_type));
#endif
    
    real_type max_rel_error = 0.0;
    int all_verified = 1;
    
    for (int i = 0; i < nev; ++i) {
      real_type *xi = X + i * n;  /* i-th eigenvector */
      real_type li = lambda[i];   /* i-th eigenvalue */
      
      /* Compute Ax = A * x_i */
      real_type alpha_v = 1.0, beta_v = 0.0;
      csr_matvec(n, nnz, ia, ja, a, xi, Ax_verify, &alpha_v, &beta_v, "A");
      
      /* Compute residual: r = Ax - lambda*x */
      /* res_norm = ||Ax - lambda*x|| */
      /* x_norm = ||x|| */
      real_type res_norm = 0.0;
      real_type x_norm = nrm2(n, xi);
      
#if (CUDA || HIP)
      /* For GPU: compute on device using axpy and nrm2 */
      /* Copy xi to temp, then compute Ax - lambda*xi */
      vec_copy(n, Ax_verify, temp);  /* temp = Ax */
      real_type neg_lambda = -li;
      axpy(n, &neg_lambda, xi, temp);  /* temp = Ax - lambda*x */
      res_norm = nrm2(n, temp);
#else
      /* For CPU: directly compute the residual */
      for (int j = 0; j < n; ++j) {
        real_type diff = Ax_verify[j] - li * xi[j];
        res_norm += diff * diff;
      }
      res_norm = sqrt(res_norm);
#endif
      
      /* Relative error: ||Ax - lambda*x|| / (||x|| * |lambda|) */
      real_type abs_lambda = (li < 0) ? -li : li;
      real_type rel_error = res_norm / (x_norm * (abs_lambda > LOBPCG_EPS ? abs_lambda : 1.0));
      
      if (rel_error > max_rel_error) {
        max_rel_error = rel_error;
      }
      
      /* Check if eigenpair is valid (relative error < 10*tol) */
      int verified = (rel_error < 10.0 * tol) ? 1 : 0;
      if (!verified) all_verified = 0;
      
      printf("  Eigenpair %3d: lambda = %16.10e, ||Ax-lx||/||x||/|l| = %12.6e  [%s]\n",
             i + 1, li, rel_error, verified ? "OK" : "FAILED");
    }
    
    printf("----------------------------\n");
    printf("Max relative error: %12.6e\n", max_rel_error);
    printf("Verification: %s\n", all_verified ? "PASSED" : "FAILED");
    printf("===========================\n\n");
    
#if (CUDA || HIP)
    freeDevice(Ax_verify);
#else
    free(Ax_verify);
#endif
  }
#endif /* LOBPCG_VERIFY */
  
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
  freeDevice(d_AS_rr);
  freeDevice(d_BS_rr);
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
  free(h_Y_small);
  free(h_lock_coeff);
  cgs2_workspace_free(cgs2_ws);
#if (CUDA || HIP)
  freeDevice(d_orth_coeff);
  freeDevice(d_Y_small);
  freeDevice(d_lock_coeff);
#endif
}
