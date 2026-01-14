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
#define LOBPCG_EPS 1e-12

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
  int *locked = (int*) calloc(nev, sizeof(int));
  
#if (CUDA || HIP)
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
    /* For GPU: need to copy X and AX to do the multiplication, or use device GEMM */
    /* We'll compute on host for the small dense matrix */
    real_type *h_X_temp = (real_type *) malloc(n * k_active * sizeof(real_type));
    real_type *h_AX_temp = (real_type *) malloc(n * k_active * sizeof(real_type));
    memcpyDevice(h_X_temp, X, n * k_active, sizeof(real_type), "D2H");
    memcpyDevice(h_AX_temp, AX, n * k_active, sizeof(real_type), "D2H");
    
    /* Lambda = X^T * AX on host */
    for (int i = 0; i < k_active; ++i) {
      for (int j = 0; j < k_active; ++j) {
        real_type sum = 0.0;
        for (int l = 0; l < n; ++l) {
          sum += h_X_temp[l + i * n] * h_AX_temp[l + j * n];
        }
        h_Lambda[i + j * k_active] = sum;  /* Column-major */
      }
    }
    
    /* Diagonal elements are the eigenvalue estimates */
    for (int i = 0; i < k_active; ++i) {
      h_lambda[i] = h_Lambda[i + i * k_active];
    }
    
    free(h_X_temp);
    free(h_AX_temp);
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
    /* Residuals: R = AX - X * Lambda */
    /* ------------------------------ */
    /* R = AX - X * Lambda  where Lambda is k_active x k_active */
    /* First copy AX to R, then R = R - X * Lambda */
    for (int i = 0; i < k_active; ++i) {
      vec_copy(n, AX + i * n, R + i * n);
    }
    
#if (CUDA || HIP)
    /* Copy Lambda to device and do GEMM */
    real_type *d_Lambda_temp;
    d_Lambda_temp = (real_type *) mallocForDevice(d_Lambda_temp, k_active * k_active, sizeof(real_type));
    memcpyDevice(d_Lambda_temp, h_Lambda, k_active * k_active, sizeof(real_type), "H2D");
    
    real_type neg_one_h = -1.0, one_h2 = 1.0;
    /* R = R - X * Lambda = -1.0 * X * Lambda + 1.0 * R */
    gemm("N", "N", n, k_active, k_active, &neg_one_h, X, n, d_Lambda_temp, k_active, &one_h2, R, n);
    
    freeDevice(d_Lambda_temp);
#else
    /* CPU: R = R - X * Lambda */
    real_type neg_one_h = -1.0, one_h2 = 1.0;
    gemm("N", "N", n, k_active, k_active, &neg_one_h, X, n, h_Lambda, k_active, &one_h2, R, n);
#endif
    
    /* Compute residual norms */
    compute_col_norms(n, k_active, R, h_res_norms);
    
    /* Print iteration info */
    real_type max_res = 0.0;
    for (int i = 0; i < k_active; ++i) {
      if (h_res_norms[i] > max_res) max_res = h_res_norms[i];
    }
    printf("LOBPCG: it %4d  max residual = %.3e\n", iter, max_res);
    
    if (res_history != NULL) {
      for (int i = 0; i < k_active; ++i) {
        res_history[(iter - 1) * nev + i] = h_res_norms[i];
      }
    }
    
    /* ------------------------------ */
    /* Lock converged eigenpairs      */
    /* ------------------------------ */
    int any_locked = 0;
    for (int i = 0; i < k_active; ++i) {
      locked[i] = (h_res_norms[i] < tol) ? 1 : 0;
      if (locked[i]) {
        any_locked = 1;
        printf("LOBPCG:   Eigenpair %d converged, lambda = %.10e, res = %.3e\n", 
               n_locked + 1, h_lambda[i], h_res_norms[i]);
      }
    }
    
    /* Move locked eigenpairs - simplified: just track count for now */
    /* In full implementation, would compact active vectors */
    if (any_locked) {
      for (int i = 0; i < k_active; ++i) {
        if (locked[i]) {
          /* Copy to locked storage */
          vec_copy(n, X + i * n, X_lock + n_locked * n);
          lambda[n_locked] = h_lambda[i];
          n_locked++;
        }
      }
      
      /* Check if all converged */
      if (n_locked >= nev) {
        printf("LOBPCG: All %d eigenpairs converged\n", nev);
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
    
    /* ------------------------------ */
    /* Orthogonalize W against locked vectors */
    /* ------------------------------ */
    if (n_locked > 0) {
      for (int i = 0; i < k_active; ++i) {
        real_type *wi = W + i * n;
        for (int j = 0; j < n_locked; ++j) {
          real_type *xlj = X_lock + j * n;
          real_type alpha_orth = dot(n, xlj, wi);
          axpy(n, -alpha_orth, xlj, wi);
        }
      }
    }
    
    /* Orthogonalize W against X */
    for (int i = 0; i < k_active; ++i) {
      real_type *wi = W + i * n;
      for (int j = 0; j < k_active; ++j) {
        real_type *xj = X + j * n;
        real_type alpha_orth = dot(n, xj, wi);
        axpy(n, -alpha_orth, xj, wi);
      }
    }
    
    /* Orthogonalize W against P (if we have P) */
    if (has_P) {
      for (int i = 0; i < k_active; ++i) {
        real_type *wi = W + i * n;
        for (int j = 0; j < k_active; ++j) {
          real_type *pj = P + j * n;
          real_type alpha_orth = dot(n, pj, wi);
          axpy(n, -alpha_orth, pj, wi);
        }
      }
    }
    
    /* Drop tiny W directions */
    compute_col_norms(n, k_active, W, h_res_norms);
    int kW = 0;
    for (int i = 0; i < k_active; ++i) {
      if (h_res_norms[i] > LOBPCG_EPS) {
        if (kW != i) {
          vec_copy(n, W + i * n, W + kW * n);
        }
        kW++;
      }
    }
    
    /* Drop tiny P directions */
    int kP = 0;
    if (has_P) {
      compute_col_norms(n, k_active, P, h_res_norms);
      for (int i = 0; i < k_active; ++i) {
        if (h_res_norms[i] > LOBPCG_EPS) {
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
    int ssize = kX + kW + kP;  /* Total subspace size */
    
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
    /* Compute small dense matrices for Rayleigh-Ritz */
    /* AS_small = S' * (A*S), BS_small = S' * S */
    /* ------------------------------ */
    /* These are ssize x ssize dense matrices in column-major order */
    /* AS[i,j] = S[:,i]' * (A * S[:,j]) stored at h_AS[i + j * ssize] */
    for (int i = 0; i < ssize; ++i) {
      for (int j = 0; j < ssize; ++j) {
        h_AS[i + j * ssize] = dot(n, S + i * n, AS_temp + j * n);
        h_BS[i + j * ssize] = dot(n, S + i * n, S + j * n);
      }
    }
    
    /* Symmetrize AS = (AS + AS') / 2 (column-major) */
    for (int i = 0; i < ssize; ++i) {
      for (int j = i + 1; j < ssize; ++j) {
        real_type avg = 0.5 * (h_AS[i + j * ssize] + h_AS[j + i * ssize]);
        h_AS[i + j * ssize] = avg;
        h_AS[j + i * ssize] = avg;
      }
    }
    
    /* ------------------------------ */
    /* Solve generalized eigenvalue problem: AS * y = theta * BS * y */
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
    
    /* Zero out Xnew and Pnew */
    vec_zero(n * k_active, Xnew);
    vec_zero(n * k_active, Pnew);
    
    for (int j = 0; j < k_active; ++j) {
      /* Accumulate Xnew[:,j] = sum_i X[:,i] * Y[i,j] + sum_i W[:,i] * Y[kX+i,j] + ... */
      /* h_Y is ssize x ssize in column-major: Y[i,j] = h_Y[i + j * ssize] */
      
      /* Contribution from X */
      for (int i = 0; i < kX; ++i) {
        real_type coeff = h_Y[i + j * ssize];  /* Y[i,j] in column-major */
        axpy(n, coeff, X + i * n, Xnew + j * n);
      }
      
      /* Contribution from W */
      for (int i = 0; i < kW; ++i) {
        real_type coeff = h_Y[(kX + i) + j * ssize];
        axpy(n, coeff, W + i * n, Xnew + j * n);
        axpy(n, coeff, W + i * n, Pnew + j * n);
      }
      
      /* Contribution from P */
      for (int i = 0; i < kP; ++i) {
        real_type coeff = h_Y[(kX + kW + i) + j * ssize];
        axpy(n, coeff, P + i * n, Xnew + j * n);
        axpy(n, coeff, P + i * n, Pnew + j * n);
      }
    }
    
    /* Orthonormalize Xnew */
    cgs2(n, k_active, Xnew);
    
    /* Stabilize Pnew: orthogonalize against Xnew */
    for (int i = 0; i < k_active; ++i) {
      real_type *pnew_i = Pnew + i * n;
      for (int j = 0; j < k_active; ++j) {
        real_type *xnew_j = Xnew + j * n;
        real_type alpha_orth = dot(n, xnew_j, pnew_i);
        axpy(n, -alpha_orth, xnew_j, pnew_i);
      }
    }
    
    /* Orthonormalize Pnew */
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
}
