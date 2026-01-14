#include <rocsparse.h>
#include <rocblas.h>
#include <hip/hip_runtime_api.h>
#include "hip_blas.h"

// Error checking macros
#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      fprintf(stderr, "HIP error in %s at line %d: %s\n", \
              __FILE__, __LINE__, hipGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define ROCBLAS_CHECK(call) \
  do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
      fprintf(stderr, "rocBLAS error in %s at line %d: %d\n", \
              __FILE__, __LINE__, status); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define ROCSPARSE_CHECK(call) \
  do { \
    rocsparse_status status = call; \
    if (status != rocsparse_status_success) { \
      fprintf(stderr, "rocSPARSE error in %s at line %d: %d\n", \
              __FILE__, __LINE__, status); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)


static rocblas_handle handle_rocblas;
static rocsparse_handle handle_rocsparse;
static void *mv_buffer = NULL;
static void *L_buffer;
static void *U_buffer;
static void *ichol_buffer;

static rocsparse_mat_descr matA = NULL;
static rocsparse_mat_descr descrL, descrU, descrA;
static rocsparse_mat_descr descrLt; //for ICHOL
static rocsparse_mat_info infoL, infoU, infoLic, infoLtic;
static rocsparse_mat_info infoA;
static rocsparse_mat_descr descrLic, descrLtic, descrM; //ICHOL
static rocsparse_mat_info infoM; //ICHOL


void initialize_handles() {

  ROCBLAS_CHECK(rocblas_create_handle(&handle_rocblas));
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle_rocsparse));

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&(descrL)));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero));

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&(descrU)));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrU, rocsparse_index_base_zero));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper));

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&(descrA)));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general));

  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoA));
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoL));
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoU));
  HIP_CHECK(hipDeviceSynchronize());
}

void analyze_spmv(const int n,
                  const int nnz,
                  int *ia,
                  int *ja,
                  real_type *a,
                  const real_type *x,
                  real_type *result,
                  char *option) {
  /* no buffer in matvec */
  if (strcmp(option, "A") == 0) {
    ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(handle_rocsparse,
                                              rocsparse_operation_none,
                                              n,
                                              n,
                                              nnz,
                                              descrA,
                                              a,
                                              ia,
                                              ja,
                                              infoA));
  }

  if (strcmp(option, "L") == 0) {
    ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(handle_rocsparse,
                                              rocsparse_operation_none,
                                              n,
                                              n,
                                              nnz,
                                              descrL,
                                              a,
                                              ia,
                                              ja,
                                              infoL));
  }

  if (strcmp(option, "U") == 0) {
    ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(handle_rocsparse,
                                              rocsparse_operation_none,
                                              n,
                                              n,
                                              nnz,
                                              descrU,
                                              a,
                                              ia,
                                              ja,
                                              infoU));
  }

  HIP_CHECK(hipDeviceSynchronize());
}

void initialize_and_analyze_L_and_U_solve(const int n,
                                          const int nnzL,
                                          int *lia,
                                          int *lja,
                                          real_type *la,
                                          const int nnzU,
                                          int *uia,
                                          int *uja,
                                          real_type *ua) {

  size_t L_buffer_size;
  size_t U_buffer_size;

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                                               rocsparse_operation_none,
                                               n,
                                               nnzL,
                                               descrL,
                                               la,
                                               lia,
                                               lja,
                                               infoL,
                                               &L_buffer_size));

  HIP_CHECK(hipMalloc((void **)&(L_buffer), L_buffer_size));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                                               rocsparse_operation_none,
                                               n,
                                               nnzU,
                                               descrU,
                                               ua,
                                               uia,
                                               uja,
                                               infoU,
                                               &U_buffer_size));
  HIP_CHECK(hipMalloc((void **)&(U_buffer), U_buffer_size));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
                                            rocsparse_operation_none,
                                            n,
                                            nnzL,
                                            descrL,
                                            la,
                                            lia,
                                            lja,
                                            infoL,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            L_buffer));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
                                            rocsparse_operation_none,
                                            n,
                                            nnzU,
                                            descrU,
                                            ua,
                                            uia,
                                            uja,
                                            infoU,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            U_buffer));
  HIP_CHECK(hipDeviceSynchronize());
}

void initialize_ichol(const int n,
                      const int nnzA,
                      int *ia,
                      int *ja,
                      real_type *a) {
  // printf("initializing ICHOLi, n = %d, nnzA = %d \n",n,nnzA);
  /* Create matrix descriptor for M */
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrM));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(descrM, rocsparse_matrix_type_general));

  /* Create matrix descriptor for L */
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrLic));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrLic, rocsparse_fill_mode_lower));
  ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descrLic, rocsparse_diag_type_non_unit));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrLic, rocsparse_index_base_zero));

  /* Create matrix descriptor for L' */
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrLtic));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrLtic, rocsparse_fill_mode_upper));
  ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descrLtic, rocsparse_diag_type_non_unit));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrLtic, rocsparse_index_base_zero));

  /* Create matrix info structure */
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoM));
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoLic));
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoLtic));

  /* Obtain required buffer size */
  size_t buffer_size_M;
  size_t buffer_size_L;
  size_t buffer_size_Lt;

  ROCSPARSE_CHECK(rocsparse_dcsric0_buffer_size(handle_rocsparse,
                                                n,
                                                nnzA,
                                                descrM,
                                                a,
                                                ia,
                                                ja,
                                                infoM,
                                                &buffer_size_M));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                                               rocsparse_operation_none,
                                               n,
                                               nnzA,
                                               descrLic,
                                               a,
                                               ia,
                                               ja,
                                               infoM,
                                               &buffer_size_L));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                                               rocsparse_operation_transpose,
                                               n,
                                               nnzA,
                                               descrLic,
                                               a,
                                               ia,
                                               ja,
                                               infoM,
                                               &buffer_size_Lt));
  // printf("Buffer sizes: %d %d %d \n", buffer_size_M, buffer_size_L, buffer_size_Lt);
  size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));
  // printf("finalsize %d \n",buffer_size);
  // Allocate temporary buffer
  HIP_CHECK(hipMalloc(&ichol_buffer, buffer_size));

  /* Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
   * computation performance */
  ROCSPARSE_CHECK(rocsparse_dcsric0_analysis(handle_rocsparse,
                                             n,
                                             nnzA,
                                             descrM,
                                             a,
                                             ia,
                                             ja,
                                             infoM,
                                             rocsparse_analysis_policy_reuse,
                                             rocsparse_solve_policy_auto,
                                             ichol_buffer));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
                                            rocsparse_operation_none,
                                            n,
                                            nnzA,
                                            descrLic,
                                            a,
                                            ia,
                                            ja,
                                            infoM,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            ichol_buffer));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
                                            rocsparse_operation_transpose,
                                            n,
                                            nnzA,
                                            descrLic,
                                            a,
                                            ia,
                                            ja,
                                            infoM,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            ichol_buffer));

  /* Check for zero pivot */
  rocsparse_int position;
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
                                                                 infoM,
                                                                 &position)) {
    printf("A has structural zero at A(%d,%d)\n", position, position);
  }

  /* Compute incomplete Cholesky factorization M = LL' */
  ROCSPARSE_CHECK(rocsparse_dcsric0(handle_rocsparse,
                                    n,
                                    nnzA,
                                    descrM,
                                    a,
                                    ia,
                                    ja,
                                    infoM,
                                    rocsparse_solve_policy_auto,
                                    ichol_buffer));

  /* Check for zero pivot */
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
                                                                 infoM,
                                                                 &position)) {
    printf("L has structural and/or numerical zero at L(%d,%d)\n",
           position,
           position);
  }
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_ichol(const int *ia,
               const int *ja,
               real_type *a,
               const int nnzA,
               pdata *prec_data,
               real_type *x,
               real_type *y) {
  real_type one = 1.0;
  ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle_rocsparse,
                                         rocsparse_operation_none,
                                         prec_data->n,
                                         nnzA,
                                         &one,
                                         descrLic,
                                         prec_data->ichol_vals,
                                         ia,
                                         ja,
                                         infoM,
                                         x,       //input
                                         prec_data->aux_vec1, //output
                                         rocsparse_solve_policy_auto,
                                         ichol_buffer));

  /* Solve L'y = z */
  ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle_rocsparse,
                                         rocsparse_operation_transpose,
                                         prec_data->n,
                                         nnzA,
                                         &one,
                                         descrLic,
                                         prec_data->ichol_vals,
                                         ia,
                                         ja,
                                         infoM,
                                         prec_data->aux_vec1,
                                         y,
                                         rocsparse_solve_policy_auto,
                                         ichol_buffer));
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void hip_vec_vec_kernel(const int n,
                                   const real_type *x,
                                   const real_type *y,
                                   real_type *z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    z[idx] = x[idx] * y[idx];

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_reciprocal_kernel(const int n,
                                          const real_type *x,
                                          real_type *z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    if (x[idx] != 0.0) {
      z[idx] = 1.0 / x[idx];
    } else {
      z[idx] = 0.0;
    }

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_sqrt_kernel(const int n,
                                    const real_type *x,
                                    real_type *z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    if (x[idx] > 0) {
      z[idx] = sqrt(x[idx]);
    } else {
      z[idx] = 0.0;
    }

    idx += blockDim.x * gridDim.x;
  }
}


__global__ void hip_vec_set_kernel(const int n,
                                   real_type value,
                                   real_type *x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    x[idx] = value;

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_zero_kernel(const int n,
                                    real_type *x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    x[idx] = 0.0;

    idx += blockDim.x * gridDim.x;
  }
}

real_type hip_dot(const int n,
                  const real_type *v,
                  const real_type *w) {
  real_type sum;

  HIP_CHECK(hipDeviceSynchronize());
  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                             n,
                             v,
                             1,
                             w,
                             1,
                             &sum));
  HIP_CHECK(hipDeviceSynchronize());
  return sum;
}

void hip_scal(const int n,
              const real_type alpha,
              real_type *v) {
  ROCBLAS_CHECK(rocblas_dscal(handle_rocblas,
                              n,
                              &alpha,
                              v,
                              1));
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_axpy(const int n,
              const real_type alpha,
              const real_type *x,
              real_type *y) {
  ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                              n,
                              &alpha,
                              x,
                              1,
                              y,
                              1));
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_csr_matvec(const int n,
                    const int nnz,
                    const int *ia,
                    const int *ja,
                    const real_type *a,
                    const real_type *x,
                    real_type *result,
                    const real_type *al,
                    const real_type *bet,
                    const char *kind) {
  /* y = alpha * A * x + beta * y */
  HIP_CHECK(hipDeviceSynchronize());
  if (strcmp(kind, "A") == 0) {
    ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
                                     rocsparse_operation_none,
                                     n,
                                     n,
                                     nnz,
                                     al,
                                     descrA,
                                     a,
                                     ia,
                                     ja,
                                     infoA,
                                     x,
                                     bet,
                                     result));
  }

  if (strcmp(kind, "L") == 0) {
    ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
                                     rocsparse_operation_none,
                                     n,
                                     n,
                                     nnz,
                                     al,
                                     descrL,
                                     a,
                                     ia,
                                     ja,
                                     infoL,
                                     x,
                                     bet,
                                     result));
  }

  if (strcmp(kind, "U") == 0) {
    ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
                                     rocsparse_operation_none,
                                     n,
                                     n,
                                     nnz,
                                     al,
                                     descrU,
                                     a,
                                     ia,
                                     ja,
                                     infoU,
                                     x,
                                     bet,
                                     result));
  }
  HIP_CHECK(hipDeviceSynchronize());
}


void hip_gemv(const char *T,
              const int m,
              const int n,
              const double *alpha,
              const double *A,
              const int lda,
              const double *x,
              const double *beta,
              double *y) {
  rocblas_operation op;
  if (strcmp(T, "T") == 0) {
    //transpose
    op = rocblas_operation_transpose;
  } else {
    //non-transpose, default
    op = rocblas_operation_none;
  }

  ROCBLAS_CHECK(rocblas_dgemv(handle_rocblas,
                              op,
                              m,
                              n,
                              alpha,
                              A,
                              lda,
                              x,
                              1,
                              beta,
                              y,
                              1));
}


void hip_lower_triangular_solve(const int n,
                                const int nnzL,
                                const int *lia,
                                const int *lja,
                                const real_type *la,
                                const real_type *diagonal,
                                const real_type *x,
                                real_type *result) {
  /* compute result = L^{-1}x */
  /* we DO NOT assume anything about L diagonal */
  /* d_x3 = L^(-1)dx2 */
  real_type one = 1.0;

  HIP_CHECK(hipDeviceSynchronize());
  ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle_rocsparse,
                                         rocsparse_operation_none,
                                         n,
                                         nnzL,
                                         &one,
                                         descrL,
                                         la,
                                         lia,
                                         lja,
                                         infoL,
                                         x,
                                         result,
                                         rocsparse_solve_policy_auto,
                                         L_buffer));
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_upper_triangular_solve(const int n,
                                const int nnzU,
                                const int *uia,
                                const int *uja,
                                const real_type *ua,
                                const real_type *diagonal,
                                const real_type *x,
                                real_type *result) {
  /* compute result = U^{-1}x */
  real_type one = 1.0;
  HIP_CHECK(hipDeviceSynchronize());
  ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle_rocsparse,
                                         rocsparse_operation_none,
                                         n,
                                         nnzU,
                                         &one,
                                         descrU,
                                         ua,
                                         uia,
                                         uja,
                                         infoU,
                                         x,
                                         result,
                                         rocsparse_solve_policy_auto,
                                         U_buffer));
  HIP_CHECK(hipDeviceSynchronize());
}

/* not std blas but needed and embarassingly parallel */

/* hip vec-vec computes an element-wise product (needed for scaling) */

void hip_vec_vec(const int n,
                 const real_type *x,
                 const real_type *y,
                 real_type *res) {
  hipLaunchKernelGGL(hip_vec_vec_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, x, y, res);
  HIP_CHECK(hipDeviceSynchronize());
}

/* vector reciprocal computes 1./d */

void hip_vector_reciprocal(const int n,
                           const real_type *v,
                           real_type *res) {
  hipLaunchKernelGGL(hip_vec_reciprocal_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, v, res);
  HIP_CHECK(hipDeviceSynchronize());
}

// vector sqrt takes an sqrt from each vector entry

void hip_vector_sqrt(const int n,
                     const real_type *v,
                     real_type *res) {
  hipLaunchKernelGGL(hip_vec_sqrt_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, v, res);
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_vec_copy(const int n,
                  const real_type *src,
                  real_type *dest) {
  HIP_CHECK(hipMemcpy(dest, src, sizeof(real_type) * n, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipDeviceSynchronize());
}


void hip_vec_set(const int n,
                 real_type value,
                 real_type *vec) {
  hipLaunchKernelGGL(hip_vec_set_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, value, vec);
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_vec_zero(const int n,
                  real_type *vec) {
  hipLaunchKernelGGL(hip_vec_zero_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, vec);
  HIP_CHECK(hipDeviceSynchronize());
}

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
              const int ldc) {
  rocblas_operation opA = (transA[0] == 'T' || transA[0] == 't') 
                           ? rocblas_operation_transpose 
                           : rocblas_operation_none;
  rocblas_operation opB = (transB[0] == 'T' || transB[0] == 't') 
                           ? rocblas_operation_transpose 
                           : rocblas_operation_none;
  
  ROCBLAS_CHECK(rocblas_dgemm(handle_rocblas,
                              opA,
                              opB,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              lda,
                              B,
                              ldb,
                              beta,
                              C,
                              ldc));
  HIP_CHECK(hipDeviceSynchronize());
}

real_type hip_nrm2(const int n, const real_type *v) {
  real_type result;
  ROCBLAS_CHECK(rocblas_dnrm2(handle_rocblas, n, v, 1, &result));
  HIP_CHECK(hipDeviceSynchronize());
  return result;
}

/* 
 * Simple Jacobi eigenvalue algorithm for symmetric matrices (host version)
 * Used for small dense matrices in LOBPCG Rayleigh-Ritz step
 */
static void hip_jacobi_eigen_host(int n, real_type *A, real_type *w, real_type *V) {
  int max_iter = 100 * n * n;
  real_type eps = 1e-14;
  
  /* Initialize V to identity */
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      V[i + j * n] = (i == j) ? 1.0 : 0.0;
    }
  }
  
  /* Work on copy of A */
  real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
  for (int i = 0; i < n * n; ++i) {
    Acopy[i] = A[i];
  }
  
  for (int iter = 0; iter < max_iter; ++iter) {
    /* Find largest off-diagonal element */
    int p = 0, q = 1;
    real_type max_val = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        real_type absval = fabs(Acopy[i + j * n]);
        if (absval > max_val) {
          max_val = absval;
          p = i;
          q = j;
        }
      }
    }
    
    if (max_val < eps) break;
    
    /* Compute Jacobi rotation */
    real_type app = Acopy[p + p * n];
    real_type aqq = Acopy[q + q * n];
    real_type apq = Acopy[p + q * n];
    
    real_type theta = 0.5 * atan2(2.0 * apq, aqq - app);
    real_type c = cos(theta);
    real_type s = sin(theta);
    
    /* Apply rotation to Acopy */
    for (int i = 0; i < n; ++i) {
      if (i != p && i != q) {
        real_type aip = Acopy[i + p * n];
        real_type aiq = Acopy[i + q * n];
        Acopy[i + p * n] = c * aip - s * aiq;
        Acopy[p + i * n] = Acopy[i + p * n];
        Acopy[i + q * n] = s * aip + c * aiq;
        Acopy[q + i * n] = Acopy[i + q * n];
      }
    }
    Acopy[p + p * n] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    Acopy[q + q * n] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    Acopy[p + q * n] = 0.0;
    Acopy[q + p * n] = 0.0;
    
    /* Apply rotation to V */
    for (int i = 0; i < n; ++i) {
      real_type vip = V[i + p * n];
      real_type viq = V[i + q * n];
      V[i + p * n] = c * vip - s * viq;
      V[i + q * n] = s * vip + c * viq;
    }
  }
  
  /* Extract eigenvalues */
  for (int i = 0; i < n; ++i) {
    w[i] = Acopy[i + i * n];
  }
  
  /* Sort eigenvalues and eigenvectors in ascending order */
  for (int i = 0; i < n - 1; ++i) {
    int min_idx = i;
    for (int j = i + 1; j < n; ++j) {
      if (w[j] < w[min_idx]) min_idx = j;
    }
    if (min_idx != i) {
      real_type tmp = w[i];
      w[i] = w[min_idx];
      w[min_idx] = tmp;
      for (int k = 0; k < n; ++k) {
        tmp = V[k + i * n];
        V[k + i * n] = V[k + min_idx * n];
        V[k + min_idx * n] = tmp;
      }
    }
  }
  
  free(Acopy);
}

static int hip_cholesky_host(int n, real_type *A) {
  for (int j = 0; j < n; ++j) {
    real_type sum = A[j + j * n];
    for (int k = 0; k < j; ++k) {
      sum -= A[j + k * n] * A[j + k * n];
    }
    if (sum <= 0.0) return -1;
    A[j + j * n] = sqrt(sum);
    
    for (int i = j + 1; i < n; ++i) {
      sum = A[i + j * n];
      for (int k = 0; k < j; ++k) {
        sum -= A[i + k * n] * A[j + k * n];
      }
      A[i + j * n] = sum / A[j + j * n];
    }
  }
  return 0;
}

/* Standard symmetric eigenvalue problem - host side for small dense matrices */
void hip_dsyev(const int n,
               real_type *A,
               real_type *w,
               real_type *eigvecs) {
  /* A, w, eigvecs are assumed to be on HOST for dense eigenvalue problems */
  hip_jacobi_eigen_host(n, A, w, eigvecs);
}

/* Generalized symmetric eigenvalue problem: A*x = lambda*B*x */
/* Host-side implementation for small dense matrices */
void hip_dsygv(const int n,
               real_type *A,
               real_type *B,
               real_type *w,
               real_type *eigvecs) {
  /* A, B, w, eigvecs are assumed to be on HOST for dense eigenvalue problems */
  
  real_type *Bcopy = (real_type*) malloc(n * n * sizeof(real_type));
  real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
  real_type *C = (real_type*) malloc(n * n * sizeof(real_type));
  
  for (int i = 0; i < n * n; ++i) {
    Bcopy[i] = B[i];
    Acopy[i] = A[i];
  }
  
  /* Cholesky: B = L*L' */
  int ret = hip_cholesky_host(n, Bcopy);
  if (ret != 0) {
    fprintf(stderr, "Warning: Cholesky failed in hip_dsygv. Using regularization.\n");
    for (int i = 0; i < n; ++i) {
      Bcopy[i + i * n] = B[i + i * n] + 1e-10;
    }
    hip_cholesky_host(n, Bcopy);
  }
  
  /* Compute C = L^{-1} * A */
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      real_type sum = Acopy[i + j * n];
      for (int k = 0; k < i; ++k) {
        sum -= Bcopy[i + k * n] * C[k + j * n];
      }
      C[i + j * n] = sum / Bcopy[i + i * n];
    }
  }
  
  /* Compute Acopy = C * L^{-T} */
  for (int j = 0; j < n; ++j) {
    for (int i = n - 1; i >= 0; --i) {
      real_type sum = C[j + i * n];
      for (int k = i + 1; k < n; ++k) {
        sum -= Bcopy[k + i * n] * Acopy[j + k * n];
      }
      Acopy[j + i * n] = sum / Bcopy[i + i * n];
    }
  }
  
  /* Symmetrize */
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      real_type avg = 0.5 * (Acopy[i + j * n] + Acopy[j + i * n]);
      Acopy[i + j * n] = avg;
      Acopy[j + i * n] = avg;
    }
  }
  
  /* Solve standard eigenvalue problem */
  hip_jacobi_eigen_host(n, Acopy, w, eigvecs);
  
  /* Back-transform eigenvectors: x = L^{-T} * y */
  for (int k = 0; k < n; ++k) {
    for (int i = n - 1; i >= 0; --i) {
      real_type sum = eigvecs[i + k * n];
      for (int j = i + 1; j < n; ++j) {
        sum -= Bcopy[j + i * n] * eigvecs[j + k * n];
      }
      eigvecs[i + k * n] = sum / Bcopy[i + i * n];
    }
  }
  
  free(Bcopy);
  free(Acopy);
  free(C);
}
