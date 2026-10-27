#include <rocsparse.h>
#include <rocsparse/rocsparse-version.h>
#include <rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>
#include <hip/hip_runtime_api.h>
#include <string.h>
#include "hip_blas.h"

// hiprand error checking
#define HIPRAND_CHECK(call) \
  do { \
    hiprandStatus_t status = call; \
    if (status != HIPRAND_STATUS_SUCCESS) { \
      printf("hiprand error %d at %s:%d\n", status, __FILE__, __LINE__); \
    } \
  } while(0)

// All sparse operations use the rocSPARSE generic API (rocsparse_v2_spmv,
// rocsparse_sptrsv, rocsparse_spic0), which requires rocsparse >= 4.6
#define ROCSPARSE_VERSION_CODE(major, minor) ((major) * 100 + (minor))
#define ROCSPARSE_VERSION_CURRENT ROCSPARSE_VERSION_CODE(ROCSPARSE_VERSION_MAJOR, ROCSPARSE_VERSION_MINOR)
#if ROCSPARSE_VERSION_CURRENT < ROCSPARSE_VERSION_CODE(4, 6)
#error "rocSPARSE 4.6 or newer is required for the generic sparse API"
#endif

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

// Generic API SpMV state, one per matrix
typedef struct {
  rocsparse_spmat_descr mat;
  rocsparse_dnvec_descr x;
  rocsparse_dnvec_descr y;
  rocsparse_spmv_descr descr;
  void *buffer;
  size_t buffer_size;
} spmv_data;

// Generic API triangular solve state, one per triangular factor
typedef struct {
  rocsparse_spmat_descr mat;
  rocsparse_dnvec_descr x;
  rocsparse_dnvec_descr y;
  rocsparse_sptrsv_descr descr;
  void *buffer;
  size_t buffer_size;
} sptrsv_data;

static spmv_data spmvA, spmvL, spmvU;
static sptrsv_data sptrsvL, sptrsvU;      // GS_std
static sptrsv_data sptrsvLic, sptrsvLtic; // ICHOL: L and L'
// rocsparse_sptrsv keeps a pointer to alpha, so alpha must outlive the descriptors
static const real_type sptrsv_alpha = 1.0;

// GPU eigensolver buffers (pre-allocated for reuse)
static real_type *d_eig_A = NULL;
static real_type *d_eig_B = NULL;
static real_type *d_eig_W = NULL;
static real_type *d_eig_E = NULL;
static rocblas_int *d_eig_info = NULL;
static int eig_max_n = 0;  // Track max size allocated


void initialize_handles() {

  ROCBLAS_CHECK(rocblas_create_handle(&handle_rocblas));
  /* Disable atomics for deterministic results (parallel reductions can vary otherwise) */
  ROCBLAS_CHECK(rocblas_set_atomics_mode(handle_rocblas, rocblas_atomics_not_allowed));
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle_rocsparse));
  HIP_CHECK(hipDeviceSynchronize());
}

static rocsparse_spmat_descr create_csr_descr(const int n,
                                              const int nnz,
                                              int *ia,
                                              int *ja,
                                              real_type *a) {
  rocsparse_spmat_descr mat;
  ROCSPARSE_CHECK(rocsparse_create_csr_descr(&mat,
                                             n,
                                             n,
                                             nnz,
                                             ia,
                                             ja,
                                             a,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_r));
  return mat;
}

static void spmv_setup(spmv_data *s,
                       const int n,
                       const int nnz,
                       int *ia,
                       int *ja,
                       real_type *a,
                       const real_type *x,
                       real_type *y,
                       const rocsparse_spmv_alg alg) {
  const rocsparse_operation op = rocsparse_operation_none;
  const rocsparse_datatype datatype = rocsparse_datatype_f64_r;
  const real_type alpha = 1.0, beta = 0.0;

  /* analyze_spmv may be called again for the same matrix kind */
  if (s->descr != NULL) {
    ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(s->descr));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(s->mat));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(s->x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(s->y));
    if (s->buffer != NULL) {
      HIP_CHECK(hipFree(s->buffer));
    }
  }

  s->mat = create_csr_descr(n, nnz, ia, ja, a);
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&s->x, n, (void*)x, datatype));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&s->y, n, (void*)y, datatype));

  ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&s->descr));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse, s->descr, rocsparse_spmv_input_alg,
                                           &alg, sizeof(alg), NULL));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse, s->descr, rocsparse_spmv_input_operation,
                                           &op, sizeof(op), NULL));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse, s->descr, rocsparse_spmv_input_scalar_datatype,
                                           &datatype, sizeof(datatype), NULL));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse, s->descr, rocsparse_spmv_input_compute_datatype,
                                           &datatype, sizeof(datatype), NULL));

  /* The analysis buffer is only needed during analysis; the results are kept in the descriptor */
  size_t analysis_buffer_size;
  void *analysis_buffer = NULL;
  ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle_rocsparse,
                                                s->descr,
                                                s->mat,
                                                s->x,
                                                s->y,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &analysis_buffer_size,
                                                NULL));
  if (analysis_buffer_size > 0) {
    HIP_CHECK(hipMalloc(&analysis_buffer, analysis_buffer_size));
  }
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    s->descr,
                                    &alpha,
                                    s->mat,
                                    s->x,
                                    &beta,
                                    s->y,
                                    rocsparse_v2_spmv_stage_analysis,
                                    analysis_buffer_size,
                                    analysis_buffer,
                                    NULL));
  if (analysis_buffer != NULL) {
    HIP_CHECK(hipFree(analysis_buffer));
  }

  ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle_rocsparse,
                                                s->descr,
                                                s->mat,
                                                s->x,
                                                s->y,
                                                rocsparse_v2_spmv_stage_compute,
                                                &s->buffer_size,
                                                NULL));
  s->buffer = NULL;
  if (s->buffer_size > 0) {
    HIP_CHECK(hipMalloc(&s->buffer, s->buffer_size));
  }
}

static void spmv_compute(spmv_data *s,
                         const real_type *alpha,
                         const real_type *x,
                         const real_type *beta,
                         real_type *y) {
  ROCSPARSE_CHECK(rocsparse_dnvec_set_values(s->x, (void*)x));
  ROCSPARSE_CHECK(rocsparse_dnvec_set_values(s->y, (void*)y));
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    s->descr,
                                    alpha,
                                    s->mat,
                                    s->x,
                                    beta,
                                    s->y,
                                    rocsparse_v2_spmv_stage_compute,
                                    s->buffer_size,
                                    s->buffer,
                                    NULL));
}

static void sptrsv_setup(sptrsv_data *s,
                         const int n,
                         const int nnz,
                         int *ia,
                         int *ja,
                         real_type *a,
                         const rocsparse_fill_mode fill_mode,
                         const rocsparse_operation op) {
  const rocsparse_diag_type diag_type = rocsparse_diag_type_non_unit;
  const rocsparse_sptrsv_alg alg = rocsparse_sptrsv_alg_default;
  const rocsparse_datatype datatype = rocsparse_datatype_f64_r;
  const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;

  s->mat = create_csr_descr(n, nnz, ia, ja, a);
  ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(s->mat, rocsparse_spmat_fill_mode,
                                                &fill_mode, sizeof(fill_mode)));
  ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(s->mat, rocsparse_spmat_diag_type,
                                                &diag_type, sizeof(diag_type)));

  /* Vector descriptors require a valid pointer; sptrsv_solve points them at the actual vectors */
  real_type *tmp;
  HIP_CHECK(hipMalloc((void **)&tmp, n * sizeof(real_type)));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&s->x, n, tmp, datatype));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&s->y, n, tmp, datatype));

  ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&s->descr));
  ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle_rocsparse, s->descr, rocsparse_sptrsv_input_alg,
                                             &alg, sizeof(alg), NULL));
  ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle_rocsparse, s->descr, rocsparse_sptrsv_input_operation,
                                             &op, sizeof(op), NULL));
  ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle_rocsparse, s->descr, rocsparse_sptrsv_input_scalar_datatype,
                                             &datatype, sizeof(datatype), NULL));
  ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle_rocsparse, s->descr, rocsparse_sptrsv_input_compute_datatype,
                                             &datatype, sizeof(datatype), NULL));
  ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle_rocsparse, s->descr, rocsparse_sptrsv_input_analysis_policy,
                                             &policy, sizeof(policy), NULL));
  ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle_rocsparse, s->descr, rocsparse_sptrsv_input_scalar_alpha,
                                             &sptrsv_alpha, sizeof(&sptrsv_alpha), NULL));

  /* The analysis buffer is only needed during analysis; the results are kept in the descriptor */
  size_t analysis_buffer_size;
  void *analysis_buffer = NULL;
  ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle_rocsparse,
                                               s->descr,
                                               s->mat,
                                               s->x,
                                               s->y,
                                               rocsparse_sptrsv_stage_analysis,
                                               &analysis_buffer_size,
                                               NULL));
  if (analysis_buffer_size > 0) {
    HIP_CHECK(hipMalloc(&analysis_buffer, analysis_buffer_size));
  }
  ROCSPARSE_CHECK(rocsparse_sptrsv(handle_rocsparse,
                                   s->descr,
                                   s->mat,
                                   s->x,
                                   s->y,
                                   rocsparse_sptrsv_stage_analysis,
                                   analysis_buffer_size,
                                   analysis_buffer,
                                   NULL));
  if (analysis_buffer != NULL) {
    HIP_CHECK(hipFree(analysis_buffer));
  }

  ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle_rocsparse,
                                               s->descr,
                                               s->mat,
                                               s->x,
                                               s->y,
                                               rocsparse_sptrsv_stage_compute,
                                               &s->buffer_size,
                                               NULL));
  s->buffer = NULL;
  if (s->buffer_size > 0) {
    HIP_CHECK(hipMalloc(&s->buffer, s->buffer_size));
  }
  HIP_CHECK(hipFree(tmp));
}

static void sptrsv_solve(sptrsv_data *s,
                         const real_type *x,
                         real_type *y) {
  ROCSPARSE_CHECK(rocsparse_dnvec_set_values(s->x, (void*)x));
  ROCSPARSE_CHECK(rocsparse_dnvec_set_values(s->y, (void*)y));
  ROCSPARSE_CHECK(rocsparse_sptrsv(handle_rocsparse,
                                   s->descr,
                                   s->mat,
                                   s->x,
                                   s->y,
                                   rocsparse_sptrsv_stage_compute,
                                   s->buffer_size,
                                   s->buffer,
                                   NULL));
}

void analyze_spmv(const int n,
                  const int nnz,
                  int *ia,
                  int *ja,
                  real_type *a,
                  const real_type *x,
                  real_type *result,
                  char *option) {
  if (strcmp(option, "A") == 0) {
    spmv_setup(&spmvA, n, nnz, ia, ja, a, x, result, rocsparse_spmv_alg_default);
  }

  if (strcmp(option, "L") == 0) {
    spmv_setup(&spmvL, n, nnz, ia, ja, a, x, result, rocsparse_spmv_alg_default);
  }

  if (strcmp(option, "U") == 0) {
    spmv_setup(&spmvU, n, nnz, ia, ja, a, x, result, rocsparse_spmv_alg_default);
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
  sptrsv_setup(&sptrsvL, n, nnzL, lia, lja, la, rocsparse_fill_mode_lower, rocsparse_operation_none);
  sptrsv_setup(&sptrsvU, n, nnzU, uia, uja, ua, rocsparse_fill_mode_upper, rocsparse_operation_none);
  HIP_CHECK(hipDeviceSynchronize());
}

void initialize_ichol(const int n,
                      const int nnzA,
                      int *ia,
                      int *ja,
                      real_type *a) {
  // printf("initializing ICHOLi, n = %d, nnzA = %d \n",n,nnzA);
  const rocsparse_spic0_alg alg = rocsparse_spic0_alg_default;
  const rocsparse_datatype datatype = rocsparse_datatype_f64_r;
  const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;

  /* spic0 factorizes M = A in place: L overwrites the lower triangle of a */
  rocsparse_spmat_descr matM = create_csr_descr(n, nnzA, ia, ja, a);

  rocsparse_spic0_descr spic0_descr;
  ROCSPARSE_CHECK(rocsparse_spic0_descr_create(handle_rocsparse, &spic0_descr, NULL));
  ROCSPARSE_CHECK(rocsparse_spic0_set_input(handle_rocsparse, spic0_descr, rocsparse_spic0_input_alg,
                                            &alg, sizeof(alg), NULL));
  ROCSPARSE_CHECK(rocsparse_spic0_set_input(handle_rocsparse, spic0_descr, rocsparse_spic0_input_compute_datatype,
                                            &datatype, sizeof(datatype), NULL));
  ROCSPARSE_CHECK(rocsparse_spic0_set_input(handle_rocsparse, spic0_descr, rocsparse_spic0_input_analysis_policy,
                                            &policy, sizeof(policy), NULL));

  size_t buffer_size;
  void *buffer = NULL;
  rocsparse_singularity singularity;
  int64_t position;

  /* Perform analysis step */
  ROCSPARSE_CHECK(rocsparse_spic0_buffer_size(handle_rocsparse,
                                              spic0_descr,
                                              matM,
                                              matM,
                                              rocsparse_spic0_stage_analysis,
                                              &buffer_size,
                                              NULL));
  if (buffer_size > 0) {
    HIP_CHECK(hipMalloc(&buffer, buffer_size));
  }
  ROCSPARSE_CHECK(rocsparse_spic0(handle_rocsparse,
                                  spic0_descr,
                                  matM,
                                  matM,
                                  rocsparse_spic0_stage_analysis,
                                  buffer_size,
                                  buffer,
                                  NULL));

  /* Check for zero pivot */
  ROCSPARSE_CHECK(rocsparse_spic0_get_output(handle_rocsparse, spic0_descr, rocsparse_spic0_output_singularity,
                                             &singularity, sizeof(singularity), NULL));
  ROCSPARSE_CHECK(rocsparse_spic0_get_output(handle_rocsparse, spic0_descr, rocsparse_spic0_output_singularity_position,
                                             &position, sizeof(position), NULL));
  HIP_CHECK(hipDeviceSynchronize());
  /* Report zero pivots only (structural or exact), not near-zero ones */
  if (singularity == rocsparse_singularity_symbolic || singularity == rocsparse_singularity_numeric_exact) {
    printf("A has structural zero at A(%lld,%lld)\n", (long long)position, (long long)position);
  }

  /* Compute incomplete Cholesky factorization M = LL' */
  if (buffer != NULL) {
    HIP_CHECK(hipFree(buffer));
    buffer = NULL;
  }
  ROCSPARSE_CHECK(rocsparse_spic0_buffer_size(handle_rocsparse,
                                              spic0_descr,
                                              matM,
                                              matM,
                                              rocsparse_spic0_stage_compute,
                                              &buffer_size,
                                              NULL));
  if (buffer_size > 0) {
    HIP_CHECK(hipMalloc(&buffer, buffer_size));
  }
  ROCSPARSE_CHECK(rocsparse_spic0(handle_rocsparse,
                                  spic0_descr,
                                  matM,
                                  matM,
                                  rocsparse_spic0_stage_compute,
                                  buffer_size,
                                  buffer,
                                  NULL));

  /* Check for zero pivot */
  ROCSPARSE_CHECK(rocsparse_spic0_get_output(handle_rocsparse, spic0_descr, rocsparse_spic0_output_singularity,
                                             &singularity, sizeof(singularity), NULL));
  ROCSPARSE_CHECK(rocsparse_spic0_get_output(handle_rocsparse, spic0_descr, rocsparse_spic0_output_singularity_position,
                                             &position, sizeof(position), NULL));
  HIP_CHECK(hipDeviceSynchronize());
  if (singularity == rocsparse_singularity_symbolic || singularity == rocsparse_singularity_numeric_exact) {
    printf("L has structural and/or numerical zero at L(%lld,%lld)\n",
           (long long)position,
           (long long)position);
  }

  if (buffer != NULL) {
    HIP_CHECK(hipFree(buffer));
  }
  ROCSPARSE_CHECK(rocsparse_spic0_descr_destroy(handle_rocsparse, spic0_descr, NULL));
  ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matM));

  /* L' is solved as the transpose of L, the lower triangle of a */
  sptrsv_setup(&sptrsvLic, n, nnzA, ia, ja, a, rocsparse_fill_mode_lower, rocsparse_operation_none);
  sptrsv_setup(&sptrsvLtic, n, nnzA, ia, ja, a, rocsparse_fill_mode_lower, rocsparse_operation_transpose);
  HIP_CHECK(hipDeviceSynchronize());
}

void hip_ichol(const int *ia,
               const int *ja,
               real_type *a,
               const int nnzA,
               pdata *prec_data,
               real_type *x,
               real_type *y) {
  sptrsv_solve(&sptrsvLic, x, prec_data->aux_vec1);

  /* Solve L'y = z */
  sptrsv_solve(&sptrsvLtic, prec_data->aux_vec1, y);
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

  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                             n,
                             v,
                             1,
                             w,
                             1,
                             &sum));
  HIP_CHECK(hipDeviceSynchronize());  /* Sync needed to read result */
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
  /* No sync needed - async execution */
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
  /* No sync needed - async execution */
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
  spmv_data *s = NULL;
  if (strcmp(kind, "A") == 0) {
    s = &spmvA;
  }
  if (strcmp(kind, "L") == 0) {
    s = &spmvL;
  }
  if (strcmp(kind, "U") == 0) {
    s = &spmvU;
  }
  if (s == NULL) {
    return;
  }

  /* Not analyzed by analyze_spmv: use row-split, as legacy csrmv did without analysis */
  if (s->descr == NULL) {
    spmv_setup(s, n, nnz, (int*)ia, (int*)ja, (real_type*)a, x, result, rocsparse_spmv_alg_csr_rowsplit);
  }
  spmv_compute(s, al, x, bet, result);
  /* No sync needed - async execution */
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
  /* No sync needed - async execution */
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
  sptrsv_solve(&sptrsvL, x, result);
  /* No sync - async execution, next op will wait on stream */
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
  sptrsv_solve(&sptrsvU, x, result);
  /* No sync - async execution, next op will wait on stream */
}

/* not std blas but needed and embarassingly parallel */

/* hip vec-vec computes an element-wise product (needed for scaling) */

void hip_vec_vec(const int n,
                 const real_type *x,
                 const real_type *y,
                 real_type *res) {
  hipLaunchKernelGGL(hip_vec_vec_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, x, y, res);
  /* No sync needed - async execution */
}

/* vector reciprocal computes 1./d */

void hip_vector_reciprocal(const int n,
                           const real_type *v,
                           real_type *res) {
  hipLaunchKernelGGL(hip_vec_reciprocal_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, v, res);
  /* No sync needed - async execution */
}

// vector sqrt takes an sqrt from each vector entry

void hip_vector_sqrt(const int n,
                     const real_type *v,
                     real_type *res) {
  hipLaunchKernelGGL(hip_vec_sqrt_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, v, res);
  /* No sync needed - async execution */
}

void hip_vec_copy(const int n,
                  const real_type *src,
                  real_type *dest) {
  if (n <= 0 || src == NULL || dest == NULL) {
    return;
  }
  HIP_CHECK(hipMemcpy(dest, src, sizeof(real_type) * n, hipMemcpyDeviceToDevice));
  /* No sync needed for D2D - async execution */
}


void hip_vec_set(const int n,
                 real_type value,
                 real_type *vec) {
  hipLaunchKernelGGL(hip_vec_set_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, value, vec);
  /* No sync needed - async execution */
}

void hip_vec_zero(const int n,
                  real_type *vec) {
  hipLaunchKernelGGL(hip_vec_zero_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, vec);
  /* No sync needed - async execution */
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
  /* No sync needed - async execution */
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
/* GPU implementation using rocSOLVER */
void hip_dsygv(const int n,
               real_type *A,
               real_type *B,
               real_type *w,
               real_type *eigvecs) {
  /* A, B, w, eigvecs are assumed to be on HOST for dense eigenvalue problems */
  
  /* Allocate/reallocate device buffers if needed */
  if (n > eig_max_n) {
    if (d_eig_A != NULL) HIP_CHECK(hipFree(d_eig_A));
    if (d_eig_B != NULL) HIP_CHECK(hipFree(d_eig_B));
    if (d_eig_W != NULL) HIP_CHECK(hipFree(d_eig_W));
    if (d_eig_E != NULL) HIP_CHECK(hipFree(d_eig_E));
    if (d_eig_info != NULL) HIP_CHECK(hipFree(d_eig_info));
    
    HIP_CHECK(hipMalloc(&d_eig_A, n * n * sizeof(real_type)));
    HIP_CHECK(hipMalloc(&d_eig_B, n * n * sizeof(real_type)));
    HIP_CHECK(hipMalloc(&d_eig_W, n * sizeof(real_type)));
    HIP_CHECK(hipMalloc(&d_eig_E, n * sizeof(real_type)));
    HIP_CHECK(hipMalloc(&d_eig_info, sizeof(rocblas_int)));
    eig_max_n = n;
  }
  
  /* Regularization for ill-conditioned B matrix */
  real_type min_diag = fabs(B[0]);
  real_type trace = 0.0;
  for (int i = 0; i < n; ++i) {
    real_type d = fabs(B[i + i * n]);
    trace += d;
    if (d < min_diag) min_diag = d;
  }
  
  /* Apply regularization if needed (modify host copy before upload) */
  real_type *B_reg = NULL;
  if (min_diag < 1e-12) {
    B_reg = (real_type*) malloc(n * n * sizeof(real_type));
    memcpy(B_reg, B, n * n * sizeof(real_type));
    real_type reg = 1e-8 * (trace / n + 1.0);
    for (int i = 0; i < n; ++i) {
      B_reg[i + i * n] += reg;
    }
  }
  
  /* Copy matrices to device */
  HIP_CHECK(hipMemcpy(d_eig_A, A, n * n * sizeof(real_type), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_eig_B, B_reg ? B_reg : B, n * n * sizeof(real_type), hipMemcpyHostToDevice));
  
  if (B_reg) free(B_reg);
  
  /* Call rocSOLVER dsygv */
  /* itype = rocblas_eform_ax: A*x = lambda*B*x */
  /* evect = rocblas_evect_original: compute eigenvectors */
  /* uplo = rocblas_fill_upper: upper triangle stored */
  rocblas_status status = rocsolver_dsygv(handle_rocblas,
                                          rocblas_eform_ax,
                                          rocblas_evect_original,
                                          rocblas_fill_upper,
                                          n,
                                          d_eig_A,
                                          n,
                                          d_eig_B,
                                          n,
                                          d_eig_W,
                                          d_eig_E,
                                          d_eig_info);
  
  /* Check for errors */
  rocblas_int h_info = 0;
  HIP_CHECK(hipMemcpy(&h_info, d_eig_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
  
  if (status != rocblas_status_success || h_info != 0) {
    /* rocSOLVER failed - fall back to CPU Jacobi */
    real_type *Bcopy = (real_type*) malloc(n * n * sizeof(real_type));
    real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
    real_type *C = (real_type*) malloc(n * n * sizeof(real_type));
    
    memcpy(Bcopy, B, n * n * sizeof(real_type));
    memcpy(Acopy, A, n * n * sizeof(real_type));
    
    /* Add regularization */
    real_type reg = 1e-6 * (trace / n + 1.0);
    for (int i = 0; i < n; ++i) {
      Bcopy[i + i * n] += reg;
    }
    
    hip_cholesky_host(n, Bcopy);
    
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        real_type sum = Acopy[i + j * n];
        for (int k = 0; k < i; ++k) {
          sum -= Bcopy[i + k * n] * C[k + j * n];
        }
        C[i + j * n] = sum / Bcopy[i + i * n];
      }
    }
    
    for (int j = 0; j < n; ++j) {
      for (int i = n - 1; i >= 0; --i) {
        real_type sum = C[j + i * n];
        for (int kk = i + 1; kk < n; ++kk) {
          sum -= Bcopy[kk + i * n] * Acopy[j + kk * n];
        }
        Acopy[j + i * n] = sum / Bcopy[i + i * n];
      }
    }
    
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        real_type avg = 0.5 * (Acopy[i + j * n] + Acopy[j + i * n]);
        Acopy[i + j * n] = avg;
        Acopy[j + i * n] = avg;
      }
    }
    
    hip_jacobi_eigen_host(n, Acopy, w, eigvecs);
    
    for (int kk = 0; kk < n; ++kk) {
      for (int i = n - 1; i >= 0; --i) {
        real_type sum = eigvecs[i + kk * n];
        for (int j = i + 1; j < n; ++j) {
          sum -= Bcopy[j + i * n] * eigvecs[j + kk * n];
        }
        eigvecs[i + kk * n] = sum / Bcopy[i + i * n];
      }
    }
    
    free(Bcopy);
    free(Acopy);
    free(C);
    return;
  }
  
  /* Copy results back to host */
  /* Eigenvectors are in d_eig_A (column-major), eigenvalues in d_eig_W */
  HIP_CHECK(hipMemcpy(eigvecs, d_eig_A, n * n * sizeof(real_type), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(w, d_eig_W, n * sizeof(real_type), hipMemcpyDeviceToHost));
}

/**
 * Generate random vectors directly on GPU using hiprand
 * This matches CG_experiments behavior for reproducibility
 * d_vec: device pointer to n*nev doubles
 * seed: random seed for reproducibility
 */
void hip_generate_random_vectors(real_type *d_vec, int64_t n, int nev, unsigned long long seed) {
  hiprandGenerator_t gen;
  /* Use XORWOW for deterministic results (DEFAULT may vary between runs) */
  HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_XORWOW));
  HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, seed));
  
  /* Generate uniform doubles in [0, 1) */
  HIPRAND_CHECK(hiprandGenerateUniformDouble(gen, d_vec, n * nev));
  
  HIPRAND_CHECK(hiprandDestroyGenerator(gen));
  HIP_CHECK(hipDeviceSynchronize());
}
