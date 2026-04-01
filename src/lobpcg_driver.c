/*
 * LOBPCG Driver - Eigenvalue solver for sparse symmetric matrices
 * 
 * Usage: ./lap_lobpcg <matrix.mtx> <mode> <preconditioner> <tolerance> <maxit> <M> <K> <nev> [seed] [verbose]
 * 
 * Arguments:
 *   matrix.mtx    - Matrix file in Matrix Market format
 *   mode          - Matrix mode: "normal" (use matrix as-is) or "laplacian" (convert to graph Laplacian)
 *   preconditioner - Preconditioner type: "none", "it_jacobi", "line_jacobi", "GS_it", "GS_it2", "GS_std", "ic0"
 *   tolerance     - Convergence tolerance for residual norm
 *   maxit         - Maximum number of LOBPCG iterations
 *   M             - Outer iterations for preconditioner
 *   K             - Inner iterations for preconditioner
 *   nev           - Number of eigenvalues/eigenvectors to compute
 *   seed          - (Optional) Random seed for reproducibility (e.g., 12345)
 *   verbose       - (Optional) Verbose output: 0=final only (default), 1=per-iteration details
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "common.h"
#include "blas.h"
#include "io_utils.h"
#include "lobpcg.h"

#if CUDA
#include "cuda_blas.h"
#include "devMem.h"
#endif

#if OPENMP
#include "openmp_blas.h"
#endif

#if HIP
#include "hip_blas.h"
#include "devMem.h"
#endif

#define MAXIT 50000

/* Generate random initial guess on host */
void generate_random_initial_guess(int n, int nev, real_type *X) {
  for (int j = 0; j < nev; ++j) {
    for (int i = 0; i < n; ++i) {
      /* Random value between -1 and 1 */
      X[i + j * n] = 2.0 * ((real_type)rand() / (real_type)RAND_MAX) - 1.0;
    }
  }
}

/* Generate initial guess for Laplacian mode on host */
/* First column is the constant vector (eigenvector for lambda=0) */
/* Remaining columns are random */
void generate_laplacian_initial_guess(int n, int nev, real_type *X) {
  /* First column: constant vector [1, 1, 1, ..., 1]^T normalized */
  real_type norm_factor = 1.0 / sqrt((real_type)n);
  for (int i = 0; i < n; ++i) {
    X[i] = norm_factor;
  }
  
  /* Remaining columns: random, but orthogonalized against the first column */
  for (int j = 1; j < nev; ++j) {
    /* Generate random column */
    for (int i = 0; i < n; ++i) {
      X[i + j * n] = 2.0 * ((real_type)rand() / (real_type)RAND_MAX) - 1.0;
    }
    
    /* Subtract projection onto first column (make orthogonal to constant vector) */
    /* Since first column is normalized constant vector, projection = mean(column) * sqrt(n) */
    real_type sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += X[i + j * n];
    }
    real_type proj = sum * norm_factor;  /* dot product with normalized constant vector */
    for (int i = 0; i < n; ++i) {
      X[i + j * n] -= proj * norm_factor;
    }
  }
}

int main(int argc, char *argv[]) {
  real_type time_LOBPCG = 0.0;
  struct timeval t1, t2;
  
  if (argc < 9) {
    printf("Usage: %s <matrix.mtx> <mode> <preconditioner> <tolerance> <maxit> <M> <K> <nev> [seed] [verbose]\n", argv[0]);
    printf("  matrix.mtx     - Matrix file in Matrix Market format\n");
    printf("  mode           - Matrix mode: 'normal' or 'laplacian'\n");
    printf("  preconditioner - Preconditioner type: none, it_jacobi, line_jacobi, GS_it, GS_it2, GS_std, ic0\n");
    printf("  tolerance      - Convergence tolerance (e.g., 1e-8)\n");
    printf("  maxit          - Maximum LOBPCG iterations\n");
    printf("  M              - Outer iterations for preconditioner\n");
    printf("  K              - Inner iterations for preconditioner\n");
    printf("  nev            - Number of eigenvalues/eigenvectors to compute\n");
    printf("  seed           - (Optional) Random seed for reproducibility (e.g., 12345)\n");
    printf("  verbose        - (Optional) Verbose output: 0=final only (default), 1=per-iteration details\n");
    return 1;
  }
  
  const char *matrixFileName = argv[1];
  const char *matrixMode = argv[2];
  const char *precName = argv[3];
  real_type lobpcg_tol = atof(argv[4]);
  int lobpcg_maxit = atoi(argv[5]);
  int M = atoi(argv[6]);
  int K = atoi(argv[7]);
  int nev = atoi(argv[8]);
  
  /* Seed random number generator */
  unsigned int random_seed;
  if (argc >= 10) {
    random_seed = (unsigned int) atoi(argv[9]);
  } else {
    random_seed = (unsigned int) time(NULL);
  }
  /* Note: srand is called just before initial guess generation to ensure determinism */
  
  /* Verbose output (default: 0) */
  int verbose = 0;
  if (argc >= 11) {
    verbose = atoi(argv[10]);
  }
  
  /* Validate matrix mode */
  int use_laplacian = 0;
  if (strcmp(matrixMode, "laplacian") == 0) {
    use_laplacian = 1;
  } else if (strcmp(matrixMode, "normal") != 0) {
    printf("Error: Invalid matrix mode '%s'. Use 'normal' or 'laplacian'.\n", matrixMode);
    return 1;
  }
  
  /* Validate and normalize preconditioner name */
  /* Supported: none, it_jacobi, line_jacobi, GS_it, GS_it2, GS_std, ichol, ic0 */
  const char *validPrecs[] = {"none", "it_jacobi", "line_jacobi", "GS_it", "GS_it2", "GS_std", "ichol", "ic0"};
  int numValidPrecs = 8;
  int precValid = 0;
  for (int i = 0; i < numValidPrecs; i++) {
    if (strcmp(precName, validPrecs[i]) == 0) {
      precValid = 1;
      break;
    }
  }
  if (!precValid) {
    printf("Warning: Unrecognized preconditioner '%s'. Setting to 'none'.\n", precName);
    precName = "none";
  }
  
  /* Normalize ic0 to ichol (they are the same: incomplete Cholesky with zero fill-in) */
  if (strcmp(precName, "ic0") == 0) {
    precName = "ichol";
  }
  
  /* ic0/ichol only works with normal mode (not laplacian) */
  if (strcmp(precName, "ichol") == 0 && use_laplacian) {
    printf("Warning: ichol/ic0 preconditioner is not compatible with laplacian mode. Setting to 'none'.\n");
    precName = "none";
  }
  
  /* Read and setup matrix */
  mmatrix *A, *L, *U, *D;
  A = (mmatrix *) calloc(1, sizeof(mmatrix));
  L = (mmatrix *) calloc(1, sizeof(mmatrix));
  U = (mmatrix *) calloc(1, sizeof(mmatrix));
  D = (mmatrix *) calloc(1, sizeof(mmatrix));
  
  /* Read matrix file */
  if (use_laplacian) {
    /* Use read_adjacency_file which adds diagonal entries needed for Laplacian */
    read_adjacency_file(matrixFileName, A);
    coo_to_csr(A);
    /* Convert adjacency matrix to graph Laplacian: L = D - A */
    /* This makes 0 the smallest eigenvalue for connected graphs */
    create_L_and_split(A, L, U, D, 0);  /* 0 = unweighted */
    
  } else {
    /* Use matrix as-is */
    read_mm_file(matrixFileName, A);
    coo_to_csr(A);
    split(A, L, U, D);
  }
  
  printf("\n\n");
  printf("======================================================\n");
  printf("LOBPCG Eigenvalue Solver\n");
  printf("======================================================\n");
  printf("  Matrix file       : %s\n", matrixFileName);
  printf("  Matrix mode       : %s\n", use_laplacian ? "laplacian (L = D - A)" : "normal");
  printf("  Matrix size       : %d x %d\n", A->n, A->n);
  printf("  Matrix nnz        : %d\n", A->nnz_unpacked);
  printf("  Num eigenvalues   : %d\n", nev);
  printf("  Preconditioner    : %s\n", precName);
  printf("  Tolerance         : %2.16g\n", lobpcg_tol);
  printf("  Max iterations    : %d\n", lobpcg_maxit);
  printf("  M (outer iter)    : %d\n", M);
  printf("  K (inner iter)    : %d\n", K);
  printf("  Random seed       : %u\n", random_seed);
  printf("  Verbose           : %d\n", verbose);
  printf("======================================================\n\n");
  fflush(stdout);  /* Ensure header is printed before any HIP operations */
  
  if (lobpcg_maxit > MAXIT) {
    printf("  [WARNING] maxit cannot be larger than %d, resetting to MAX\n", MAXIT);
    lobpcg_maxit = MAXIT;
  }
  
  if (nev > A->n) {
    printf("  [WARNING] nev cannot be larger than matrix dimension %d, resetting\n", A->n);
    nev = A->n;
  }
  
  /* Setup preconditioner data */
  pdata *prec_data;
  prec_data = (pdata *) calloc(1, sizeof(pdata));
  prec_data->n = A->n;
  prec_data->prec_op = (char *) precName;
  prec_data->k = K;
  prec_data->m = M;
  prec_data->lnnz = L->nnz;
  prec_data->unnz = U->nnz;
  
  /* Allocate eigenvectors and eigenvalues */
  real_type *eigenvalues = (real_type *) calloc(nev, sizeof(real_type));
  real_type *res_history = (real_type *) calloc(lobpcg_maxit * nev, sizeof(real_type));
  
#if (CUDA || HIP)
  initialize_handles();
  
  /* Setup device arrays */
  real_type *d_X;
  real_type *d_d;
  real_type *d_A_a;
  int *d_A_ia, *d_A_ja;
  
  d_X = (real_type *) mallocForDevice(d_X, A->n * nev, sizeof(real_type));
  d_d = (real_type *) mallocForDevice(d_d, A->n, sizeof(real_type));
  
  /* Generate initial guess */
  /* Seed random number generator here (after all library init) to ensure determinism */
  srand(random_seed);
  
  if (use_laplacian) {
    /* For Laplacian, first column is constant vector (eigenvector for lambda=0) */
    /* Generate on host and copy to device */
    real_type *h_X = (real_type *) calloc(A->n * nev, sizeof(real_type));
    generate_laplacian_initial_guess(A->n, nev, h_X);
    memcpyDevice(d_X, h_X, A->n * nev, sizeof(real_type), "H2D");
    free(h_X);
  } else {
    /* Generate random vectors directly on GPU */
    /* This matches CG_experiments behavior for reproducibility */
#if HIP
    hip_generate_random_vectors(d_X, A->n, nev, (unsigned long long)random_seed);
#elif CUDA
    cuda_generate_random_vectors(d_X, A->n, nev, (unsigned long long)random_seed);
#endif
    
    /* Debug: verify random vectors are deterministic */
    if (verbose) {
      real_type h_check[5];
      memcpyDevice(h_check, d_X, 5, sizeof(real_type), "D2H");
      printf("Debug: First 5 random values: %.10f %.10f %.10f %.10f %.10f\n",
             h_check[0], h_check[1], h_check[2], h_check[3], h_check[4]);
    }
  }
  
  /* Copy diagonal to device - ensure non-zero for preconditioners */
  real_type *h_d = (real_type *) calloc(A->n, sizeof(real_type));
  for (int i = 0; i < A->n; ++i) {
    h_d[i] = D->csr_vals[i];
    /* If diagonal is zero or too small, use 1.0 to avoid division by zero */
    if (fabs(h_d[i]) < 1e-14) {
      h_d[i] = 1.0;
    }
  }
  memcpyDevice(d_d, h_d, A->n, sizeof(real_type), "H2D");
  free(h_d);
  
  /* Setup L, U for preconditioner */
#if (CUDA || HIP)
  if (strcmp(prec_data->prec_op, "GS_std") == 0) {
    /* Add diagonal to L and U for GS_std */
    int *new_L_ja = (int *) calloc(L->nnz + L->n, sizeof(int));
    int *new_U_ja = (int *) calloc(U->nnz + U->n, sizeof(int));
    real_type *new_L_a = (real_type *) calloc(L->nnz + L->n, sizeof(real_type));
    real_type *new_U_a = (real_type *) calloc(U->nnz + U->n, sizeof(real_type));
    
    int c = 0;
    for (int ii = 0; ii < L->n; ++ii) {
      for (int jj = L->csr_ia[ii]; jj < L->csr_ia[ii + 1]; ++jj) {
        new_L_ja[c] = L->csr_ja[jj];
        new_L_a[c] = L->csr_vals[jj];
        c++;
      }
      new_L_ja[c] = ii;
      new_L_a[c] = D->csr_vals[ii];
      c++;
    }
    
    c = 0;
    for (int ii = 0; ii < U->n; ++ii) {
      new_U_ja[c] = ii;
      new_U_a[c] = D->csr_vals[ii];
      c++;
      for (int jj = U->csr_ia[ii]; jj < U->csr_ia[ii + 1]; ++jj) {
        new_U_ja[c] = U->csr_ja[jj];
        new_U_a[c] = U->csr_vals[jj];
        c++;
      }
    }
    
    for (int ii = 1; ii <= A->n; ++ii) {
      L->csr_ia[ii] += ii;
      U->csr_ia[ii] += ii;
    }
    L->nnz += A->n;
    U->nnz += A->n;
    
    free(L->csr_ja);
    free(L->csr_vals);
    free(U->csr_ja);
    free(U->csr_vals);
    
    L->csr_ja = new_L_ja;
    L->csr_vals = new_L_a;
    U->csr_ja = new_U_ja;
    U->csr_vals = new_U_a;
  }
#endif
  
  prec_data->lnnz = L->nnz;
  prec_data->unnz = U->nnz;
  
  /* Only allocate L and U matrices if preconditioner needs them */
  /* "none" preconditioner doesn't need L/U, saving memory for large matrices */
  int needs_LU = (strcmp(precName, "none") != 0);
  
  if (needs_LU) {
    /* Allocate L and U matrices on device (guard against zero nnz) */
    int l_nnz = (L->nnz > 0) ? L->nnz : 1;
    int u_nnz = (U->nnz > 0) ? U->nnz : 1;
    
    prec_data->lia = (int *) mallocForDevice(prec_data->lia, A->n + 1, sizeof(int));
    prec_data->lja = (int *) mallocForDevice(prec_data->lja, l_nnz, sizeof(int));
    prec_data->la = (real_type *) mallocForDevice(prec_data->la, l_nnz, sizeof(real_type));
    
    prec_data->uia = (int *) mallocForDevice(prec_data->uia, A->n + 1, sizeof(int));
    prec_data->uja = (int *) mallocForDevice(prec_data->uja, u_nnz, sizeof(int));
    prec_data->ua = (real_type *) mallocForDevice(prec_data->ua, u_nnz, sizeof(real_type));
    
    memcpyDevice(prec_data->lia, L->csr_ia, A->n + 1, sizeof(int), "H2D");
    if (L->nnz > 0) {
      memcpyDevice(prec_data->lja, L->csr_ja, L->nnz, sizeof(int), "H2D");
      memcpyDevice(prec_data->la, L->csr_vals, L->nnz, sizeof(real_type), "H2D");
    }
    
    memcpyDevice(prec_data->uia, U->csr_ia, A->n + 1, sizeof(int), "H2D");
    if (U->nnz > 0) {
      memcpyDevice(prec_data->uja, U->csr_ja, U->nnz, sizeof(int), "H2D");
      memcpyDevice(prec_data->ua, U->csr_vals, U->nnz, sizeof(real_type), "H2D");
    }
  } else {
    /* Set to NULL for "none" preconditioner */
    prec_data->lia = NULL;
    prec_data->lja = NULL;
    prec_data->la = NULL;
    prec_data->uia = NULL;
    prec_data->uja = NULL;
    prec_data->ua = NULL;
  }
  
  prec_data->d_r = (real_type *) mallocForDevice(prec_data->d_r, A->n, sizeof(real_type));
  vector_reciprocal(A->n, d_d, prec_data->d_r);
  prec_data->d = d_d;
  
  prec_data->aux_vec1 = (real_type *) mallocForDevice(prec_data->aux_vec1, A->n, sizeof(real_type));
  prec_data->aux_vec2 = (real_type *) mallocForDevice(prec_data->aux_vec2, A->n, sizeof(real_type));
  prec_data->aux_vec3 = (real_type *) mallocForDevice(prec_data->aux_vec3, A->n, sizeof(real_type));
  
  /* Initialize aux vectors to zero */
  vec_zero(A->n, prec_data->aux_vec1);
  vec_zero(A->n, prec_data->aux_vec2);
  vec_zero(A->n, prec_data->aux_vec3);
  
  /* Copy matrix A to device */
  d_A_ia = (int *) mallocForDevice(d_A_ia, A->n + 1, sizeof(int));
  d_A_ja = (int *) mallocForDevice(d_A_ja, A->nnz_unpacked, sizeof(int));
  d_A_a = (real_type *) mallocForDevice(d_A_a, A->nnz_unpacked, sizeof(real_type));
  
  memcpyDevice(d_A_ia, A->csr_ia, A->n + 1, sizeof(int), "H2D");
  memcpyDevice(d_A_ja, A->csr_ja, A->nnz_unpacked, sizeof(int), "H2D");
  memcpyDevice(d_A_a, A->csr_vals, A->nnz_unpacked, sizeof(real_type), "H2D");
  
  /* Store original host pointers */
  int *h_A_ia = A->csr_ia;
  int *h_A_ja = A->csr_ja;
  real_type *h_A_a = A->csr_vals;
  
  A->csr_ia = d_A_ia;
  A->csr_ja = d_A_ja;
  A->csr_vals = d_A_a;
  
  real_type *X = d_X;
  
#if HIP
  /* Analyze SpMV */
  analyze_spmv(A->n, A->nnz_unpacked, A->csr_ia, A->csr_ja, A->csr_vals, d_X, d_X, "A");
  
  /* Analyze L and U for preconditioners that use them */
  if ((strcmp(prec_data->prec_op, "GS_it") == 0) || 
      (strcmp(prec_data->prec_op, "GS_it2") == 0) ||
      (strcmp(prec_data->prec_op, "it_jacobi") == 0) ||
      (strcmp(prec_data->prec_op, "line_jacobi") == 0)) {
    analyze_spmv(A->n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la, d_X, d_X, "L");
    analyze_spmv(A->n, prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua, d_X, d_X, "U");
  }
#endif

#if CUDA
  real_type one = 1.0;
  real_type minusone = -1.0;
  initialize_spmv_buffer(A->n, A->nnz_unpacked, A->csr_ia, A->csr_ja, A->csr_vals, d_X, d_X, &one, &minusone);
  
  initialize_L_and_U_descriptors(A->n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la,
                                  prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua);
#endif

  if (strcmp(prec_data->prec_op, "GS_std") == 0) {
    initialize_and_analyze_L_and_U_solve(A->n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la,
                                          prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua);
  }
  
  /* Initialize ichol preconditioner */
  if (strcmp(prec_data->prec_op, "ichol") == 0) {
    prec_data->ichol_vals = (real_type *) mallocForDevice(prec_data->ichol_vals, A->nnz_unpacked, sizeof(real_type));
    memcpyDevice(prec_data->ichol_vals, A->csr_vals, A->nnz_unpacked, sizeof(real_type), "D2D");
    
    initialize_ichol(A->n, 
                     A->nnz_unpacked, 
                     A->csr_ia, 
                     A->csr_ja, 
                     prec_data->ichol_vals);
  }
  
#else /* CPU / OpenMP */
  prec_data->lia = L->csr_ia;
  prec_data->lja = L->csr_ja;
  prec_data->la = L->csr_vals;
  
  prec_data->uia = U->csr_ia;
  prec_data->uja = U->csr_ja;
  prec_data->ua = U->csr_vals;
  
  real_type *d_inv = (real_type *) calloc(A->n, sizeof(real_type));
  real_type *d = (real_type *) calloc(A->n, sizeof(real_type));
  for (int i = 0; i < A->n; ++i) {
    d[i] = D->csr_vals[i];
    /* If diagonal is zero or too small, use 1.0 to avoid division by zero */
    if (fabs(d[i]) < 1e-14) {
      d[i] = 1.0;
    }
  }
  vector_reciprocal(A->n, d, d_inv);
  
  prec_data->d = d;
  prec_data->d_r = d_inv;
  
  prec_data->aux_vec1 = (real_type *) calloc(A->n, sizeof(real_type));
  prec_data->aux_vec2 = (real_type *) calloc(A->n, sizeof(real_type));
  prec_data->aux_vec3 = (real_type *) calloc(A->n, sizeof(real_type));
  
  /* Allocate and generate initial guess */
  real_type *X = (real_type *) calloc(A->n * nev, sizeof(real_type));
  if (use_laplacian) {
    generate_laplacian_initial_guess(A->n, nev, X);
  } else {
    generate_random_initial_guess(A->n, nev, X);
  }
#endif
  
  /* Run LOBPCG */
  int it, nconv;
  
  gettimeofday(&t1, 0);
  lobpcg(A->n,
         A->nnz_unpacked,
         A->csr_ia,
         A->csr_ja,
         A->csr_vals,
         nev,
         X,
         eigenvalues,
         lobpcg_tol,
         prec_data,
         lobpcg_maxit,
         &it,
         &nconv,
         res_history,
         verbose);
  gettimeofday(&t2, 0);
  time_LOBPCG = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  
  /* Print results */
  printf("\n======================================================\n");
  printf("LOBPCG Summary Results\n");
  printf("======================================================\n");
  printf("  Iterations        : %d\n", it);
  printf("  Converged         : %d / %d\n", nconv, nev);
  printf("  Time (seconds)    : %2.4f\n", time_LOBPCG / 1000.0);
  printf("  Mode              : %s\n", use_laplacian ? "laplacian" : "normal");
  printf("  Preconditioner    : %s\n", prec_data->prec_op);
  printf("\nComputed Eigenvalues:\n");
  for (int i = 0; i < nev; ++i) {
    printf("  lambda[%d] = %20.15e\n", i, eigenvalues[i]);
  }
  printf("======================================================\n");
  
  /* Cleanup */
  free(eigenvalues);
  free(res_history);
  
#if (CUDA || HIP)
  if (d_X) freeDevice(d_X);
  if (d_d) freeDevice(d_d);
  if (d_A_ia) freeDevice(d_A_ia);
  if (d_A_ja) freeDevice(d_A_ja);
  if (d_A_a) freeDevice(d_A_a);
  if (prec_data->lia) freeDevice(prec_data->lia);
  if (prec_data->lja) freeDevice(prec_data->lja);
  if (prec_data->la) freeDevice(prec_data->la);
  if (prec_data->uia) freeDevice(prec_data->uia);
  if (prec_data->uja) freeDevice(prec_data->uja);
  if (prec_data->ua) freeDevice(prec_data->ua);
  if (prec_data->d_r) freeDevice(prec_data->d_r);
  if (prec_data->aux_vec1) freeDevice(prec_data->aux_vec1);
  if (prec_data->aux_vec2) freeDevice(prec_data->aux_vec2);
  if (prec_data->aux_vec3) freeDevice(prec_data->aux_vec3);
  
  /* Free ichol_vals if it was allocated */
  if (strcmp(prec_data->prec_op, "ichol") == 0 && prec_data->ichol_vals) {
    freeDevice(prec_data->ichol_vals);
  }
  
  if (h_A_ia) free(h_A_ia);
  if (h_A_ja) free(h_A_ja);
  if (h_A_a) free(h_A_a);
#else
  free(X);
  free(prec_data->d);
  free(prec_data->d_r);
  free(prec_data->aux_vec1);
  free(prec_data->aux_vec2);
  free(prec_data->aux_vec3);
#endif
  
  free(prec_data);
  free(A);
  free(L);
  free(U);
  free(D);
  
  return 0;
}
