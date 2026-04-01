#include "cublas_v2.h"

#include <cusparse.h> 
#include "cuda_blas.h"
#if USE_FP64
#define cuda_data_type CUDA_R_64F
#else
#define cuda_data_type CUDA_R_32F
#endif
static cublasHandle_t handle_cublas;
static cusparseHandle_t handle_cusparse;

static void *mv_buffer;
static void *L_buffer;
static void *U_buffer;

static cusparseSpMatDescr_t matA = NULL;
static cusparseSpMatDescr_t matL = NULL;
static cusparseSpMatDescr_t matU = NULL;

static cusparseSpSVDescr_t spsvDescrL = NULL;
static cusparseSpSVDescr_t spsvDescrU = NULL;

static cusparseDnVecDescr_t vecTmpIn = NULL;
static cusparseDnVecDescr_t vecTmpOut = NULL;
static int vecTmpSize = 0; 


void initialize_handles(){
  //printf("initializing handles! \n");
  cublasCreate(&handle_cublas);
  cusparseCreate(&handle_cusparse);
}

void initialize_spmv_buffer(const int n, 
                            const int nnz, 
                            int *ia, 
                            int *ja, 
                            real_type *a, 
                            const real_type *x, 
                            real_type *result, 
                            real_type *al, 
                            real_type *bet){
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  size_t mv_buffer_size;
  cusparseStatus_t status_cusparse;

  status_cusparse = cusparseCreateDnVec(&vecX,
                                        n,
                                        (void*) x,
                                   cuda_data_type);

  // printf("matX creation status %d\n", status_cusparse);  
  status_cusparse = cusparseCreateDnVec(&vecY,
                                        n,
                                        (void *) result,
                                         cuda_data_type);

  // printf("vecY creation status %d\n", status_cusparse);  
  status_cusparse = cusparseCreateCsr(&matA,
                                      n,
                                      n,
                                      nnz,
                                      ia,
                                      ja,
                                      a,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                       cuda_data_type);

  // printf("matA creation status %d\n", status_cusparse);  
  status_cusparse = cusparseSpMV_bufferSize(handle_cusparse,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            al,
                                            matA,
                                            vecX,
                                            bet,
                                            vecY,
                                             cuda_data_type,
#if V100
                                             CUSPARSE_CSRMV_ALG2,
#else
                                             CUSPARSE_SPMV_CSR_ALG2,
#endif     
                                       &mv_buffer_size);

  cudaDeviceSynchronize();

  // printf("mv buffer size %d alpha %f beta %f status %d \n", mv_buffer_size, *al, *bet, status_cusparse);
  cudaError t = cudaMalloc( &mv_buffer, mv_buffer_size);

  if (t != 0) printf("allocated mv_buffer: is it NULL? %d, error %d \n", mv_buffer == NULL, t);

  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
}

void initialize_and_analyze_L_and_U_solve(const int n, 
                                          const int nnzL, 
                                          int *lia, 
                                          int *lja, 
                                          real_type *la,
                                          const int nnzU, 
                                          int *uia, 
                                          int *uja, 
                                          real_type *ua){
  cusparseStatus_t status;
  size_t L_buffer_size, U_buffer_size;
  real_type one = 1.0;
  cusparseFillMode_t fillLower = CUSPARSE_FILL_MODE_LOWER;
  cusparseFillMode_t fillUpper = CUSPARSE_FILL_MODE_UPPER;
  cusparseDiagType_t diagNonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT;

  cusparseCreateCsr(&matL, n, n, nnzL, lia, lja, la,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type);
  cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE,
                            &fillLower, sizeof(cusparseFillMode_t));
  cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE,
                            &diagNonUnit, sizeof(cusparseDiagType_t));

  cusparseCreateCsr(&matU, n, n, nnzU, uia, uja, ua,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type);
  cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE,
                            &fillUpper, sizeof(cusparseFillMode_t));
  cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE,
                            &diagNonUnit, sizeof(cusparseDiagType_t));

  if (vecTmpSize < n) {
    real_type *tmpBuf;
    cudaMalloc(&tmpBuf, n * sizeof(real_type));
    if (vecTmpIn) cusparseDestroyDnVec(vecTmpIn);
    if (vecTmpOut) cusparseDestroyDnVec(vecTmpOut);
    cusparseCreateDnVec(&vecTmpIn, n, tmpBuf, cuda_data_type);
    real_type *tmpBuf2;
    cudaMalloc(&tmpBuf2, n * sizeof(real_type));
    cusparseCreateDnVec(&vecTmpOut, n, tmpBuf2, cuda_data_type);
    vecTmpSize = n;
  }

  cusparseSpSV_createDescr(&spsvDescrL);
  status = cusparseSpSV_bufferSize(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &one, matL, vecTmpIn, vecTmpOut, cuda_data_type,
                                   CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &L_buffer_size);
  cudaMalloc(&L_buffer, L_buffer_size);
  status = cusparseSpSV_analysis(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, matL, vecTmpIn, vecTmpOut, cuda_data_type,
                                 CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, L_buffer);

  cusparseSpSV_createDescr(&spsvDescrU);
  status = cusparseSpSV_bufferSize(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &one, matU, vecTmpIn, vecTmpOut, cuda_data_type,
                                   CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &U_buffer_size);
  cudaMalloc(&U_buffer, U_buffer_size);
  status = cusparseSpSV_analysis(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, matU, vecTmpIn, vecTmpOut, cuda_data_type,
                                 CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, U_buffer);
}


void initialize_L_and_U_descriptors(const int n, 
                                    const int nnzL, 
                                    int *lia, 
                                    int *lja, 
                                    real_type *la,
                                    const int nnzU, 
                                    int *uia, 
                                    int *uja, 
                                    real_type *ua){
    // Matrix descriptors are now created in initialize_and_analyze_L_and_U_solve
    // This function is kept for API compatibility
}

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a)
{
  fprintf(stderr, "ERROR: Incomplete Cholesky (ichol) preconditioner is not supported in CUDA 12+.\n");
  fprintf(stderr, "The csric02/csrsv2 APIs were removed. Please use a different preconditioner.\n");
  exit(1);
}


void cuda_ichol(const int *ia, 
                const int *ja, 
                real_type *a, 
                const int nnzA,
                pdata *prec_data, 
                real_type *x, 
                real_type *y) {
  fprintf(stderr, "ERROR: Incomplete Cholesky (ichol) preconditioner is not supported in CUDA 12+.\n");
  exit(1);
}

__global__ void cuda_vec_vec_kernel(const int n,
                                    const real_type *x,
                                    const real_type *y,
                                    real_type *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    z[idx] =  x[idx]*y[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_vec_reciprocal_kernel(const int n,
                                           const real_type *x,
                                           real_type *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    if  (x[idx] != 0.0 ){
      z[idx] = 1.0/x[idx];
    } else {
      z[idx] = 0.0;
    }

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_vec_sqrt_kernel(const int n,
                                     const real_type *x,
                                     real_type *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    if (x[idx]>0) {
      z[idx] =  sqrt(x[idx]);
    } else {
      z[idx] = 0.0;
    }

    idx += blockDim.x * gridDim.x;
  }
}


__global__ void cuda_vec_zero_kernel(const int n,
                                     real_type *x){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    x[idx] =  0.0;

    idx += blockDim.x * gridDim.x;
  }
}

real_type cuda_dot (const int n, const real_type *v, const real_type *w){
  real_type sum;

  cublasStatus_t status;
#if USE_FP64
  status = cublasDdot (handle_cublas, 
                       n, 
                       v, 
                       1, 
                       w, 
                       1, 
                       &sum);
#else
  status = cublasSdot (handle_cublas, 
                       n, 
                       v, 
                       1, 
                       w, 
                       1, 
                       &sum);
#endif
  //printf("DOT product status %d\n", status);
  return sum;
}

void cuda_scal (const int n, const real_type alpha, real_type *v){
#if USE_FP64 
 cublasDscal(handle_cublas, 
              n,
              &alpha,
              v, 
              1);
#else
 cublasSscal(handle_cublas, 
              n,
              &alpha,
              v, 
              1);
#endif
}

void cuda_axpy (const int n, const real_type alpha, const real_type *x, real_type *y){

  cublasStatus_t status;
#if USE_FP64 
 status = cublasDaxpy(handle_cublas, 
                       n,
                       &alpha,
                       x, 
                       1,
                       y, 
                       1);
#else

 status = cublasSaxpy(handle_cublas, 
                       n,
                       &alpha,
                       x, 
                       1,
                       y, 
                       1);
#endif
}

void cuda_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const real_type *a, const real_type *x, real_type *result, const real_type*al, const real_type *bet){
  /* y = alpha *A* x + beta * y */ 

  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  cusparseSpMatDescr_t matCSR;
  cusparseCreateDnVec(&vecX,
                      n,
                      (void*) x,
                      cuda_data_type);

  cusparseCreateDnVec(&vecY,
                      n,
                      (void *) result,
                      cuda_data_type);

  cusparseStatus_t status_cusparse;

  status_cusparse = cusparseCreateCsr(&matCSR,
                                      n,
                                      n,
                                      nnz,
                                      (void *)ia,
                                      (void *)ja,
                                      (void *)a,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      cuda_data_type);
  // printf("before matvec: input^Tinput %5.16e, output^Toutput %5.16e alpha %f beta %f\n", cuda_dot(n, x,x), cuda_dot(n, result, result), *al, *bet);
  status_cusparse = cusparseSpMV(handle_cusparse,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 al,
                                 matCSR,
                                 vecX,
                                 bet,
                                 vecY,
                                 cuda_data_type,
#if V100
                                 CUSPARSE_CSRMV_ALG2,
#else
                                 CUSPARSE_SPMV_CSR_ALG2,
#endif     
                            mv_buffer);
  //  printf("matvec status: %d is MV BUFFER NULL? %d  is matA null? %d\n", status_cusparse, mv_buffer == NULL, matA==NULL);
  //  printf("after matvec: input^Tinput %5.16e, output^Toutput %5.16e\n", cuda_dot(n, x,x), cuda_dot(n,result, result));

  cusparseDestroySpMat(matCSR);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
}

void cuda_lower_triangular_solve(const int n,
                                 const int nnzL, 
                                 const int *lia, 
                                 const int *lja, 
                                 const real_type *la,
                                 const real_type *diagonal, 
                                 const real_type *x, real_type *result){
  real_type one = 1.0;
  cusparseDnVecDescr_t vecX, vecY;
  cusparseCreateDnVec(&vecX, n, (void*)x, cuda_data_type);
  cusparseCreateDnVec(&vecY, n, (void*)result, cuda_data_type);
  
  cusparseSpSV_solve(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, matL, vecX, vecY, cuda_data_type,
                     CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
  
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
}


void cuda_upper_triangular_solve(const int n, 
                                 const int nnzU, 
                                 const int *uia, 
                                 const int *uja, 
                                 const real_type *ua, 
                                 const real_type *diagonal, 
                                 const real_type *x, 
                                 real_type *result){
  real_type one = 1.0;
  cusparseDnVecDescr_t vecX, vecY;
  cusparseCreateDnVec(&vecX, n, (void*)x, cuda_data_type);
  cusparseCreateDnVec(&vecY, n, (void*)result, cuda_data_type);
  
  cusparseSpSV_solve(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, matU, vecX, vecY, cuda_data_type,
                     CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU);
  
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
}

/* not std blas but needed and embarassingly parallel */

/* cuda vec-vec computes an element-wise product (needed for scaling) */

void cuda_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res){

  cuda_vec_vec_kernel<<<1024, 1024>>>(n, x, y, res);
}

/* vector reciprocal computes 1./d */ 

void cuda_vector_reciprocal(const int n, const real_type *v, real_type *res){

  cuda_vec_reciprocal_kernel<<<1024, 1024>>>(n, v, res);
}

/* vector sqrt takes an sqrt from each vector entry */

void cuda_vector_sqrt(const int n, const real_type *v, real_type *res){

  cuda_vec_sqrt_kernel<<<1024, 1024>>>(n, v, res);
}

void cuda_vec_copy(const int n, const real_type *src, real_type *dest){

  cudaMemcpy(dest, src, sizeof(real_type) * n, cudaMemcpyDeviceToDevice);
}

void cuda_vec_zero(const int n, real_type *vec){

  cuda_vec_zero_kernel<<<1024, 1024>>>(n, vec);
}

__global__ void cuda_vec_set_kernel(const int n, real_type value, real_type *x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    x[idx] = value;
    idx += blockDim.x * gridDim.x;
  }
}

void cuda_vec_set(const int n, real_type value, real_type *vec) {
  cuda_vec_set_kernel<<<1024, 1024>>>(n, value, vec);
}

void cuda_gemv(const char *T,
               const int m,
               const int n,
               const double *alpha,
               const double *A,
               const int lda,
               const double *x,
               const double *beta,
               double *y) {
  cublasOperation_t op = (T[0] == 'T' || T[0] == 't') 
                          ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasDgemv(handle_cublas, op, m, n, alpha, A, lda, x, 1, beta, y, 1);
}

void cuda_gemm(const char *transA,
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
  cublasOperation_t opA = (transA[0] == 'T' || transA[0] == 't') 
                           ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = (transB[0] == 'T' || transB[0] == 't') 
                           ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasDgemm(handle_cublas, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

real_type cuda_nrm2(const int n, const real_type *v) {
  real_type result;
  cublasDnrm2(handle_cublas, n, v, 1, &result);
  cudaDeviceSynchronize();
  return result;
}

static void cuda_jacobi_eigen_host(int n, real_type *A, real_type *w, real_type *V) {
  int max_iter = 100 * n * n;
  real_type eps = 1e-14;
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      V[i + j * n] = (i == j) ? 1.0 : 0.0;
    }
  }
  
  for (int iter = 0; iter < max_iter; ++iter) {
    int p = 0, q = 1;
    real_type max_off = fabs(A[0 + 1 * n]);
    
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (fabs(A[i + j * n]) > max_off) {
          max_off = fabs(A[i + j * n]);
          p = i;
          q = j;
        }
      }
    }
    
    if (max_off < eps) break;
    
    real_type app = A[p + p * n];
    real_type aqq = A[q + q * n];
    real_type apq = A[p + q * n];
    
    real_type theta = 0.5 * atan2(2.0 * apq, aqq - app);
    real_type c = cos(theta);
    real_type s = sin(theta);
    
    for (int i = 0; i < n; ++i) {
      real_type aip = A[i + p * n];
      real_type aiq = A[i + q * n];
      A[i + p * n] = c * aip - s * aiq;
      A[i + q * n] = s * aip + c * aiq;
      
      A[p + i * n] = A[i + p * n];
      A[q + i * n] = A[i + q * n];
    }
    
    A[p + p * n] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
    A[q + q * n] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
    A[p + q * n] = 0.0;
    A[q + p * n] = 0.0;
    
    for (int i = 0; i < n; ++i) {
      real_type vip = V[i + p * n];
      real_type viq = V[i + q * n];
      V[i + p * n] = c * vip - s * viq;
      V[i + q * n] = s * vip + c * viq;
    }
  }
  
  for (int i = 0; i < n; ++i) {
    w[i] = A[i + i * n];
  }
  
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
}

void cuda_dsyev(const int n,
                real_type *A,
                real_type *w,
                real_type *eigvecs) {
  cuda_jacobi_eigen_host(n, A, w, eigvecs);
}

static int cuda_cholesky_host(int n, real_type *A) {
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

void cuda_dsygv(const int n,
                real_type *A,
                real_type *B,
                real_type *w,
                real_type *eigvecs) {
  real_type *L = (real_type *)malloc(n * n * sizeof(real_type));
  real_type *C = (real_type *)malloc(n * n * sizeof(real_type));
  
  memcpy(L, B, n * n * sizeof(real_type));
  
  int ret = cuda_cholesky_host(n, L);
  if (ret != 0) {
    for (int i = 0; i < n; ++i) {
      L[i + i * n] += 1e-10;
    }
    cuda_cholesky_host(n, L);
  }
  
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      real_type sum = A[i + j * n];
      for (int k = 0; k < i; ++k) {
        sum -= L[i + k * n] * C[k + j * n];
      }
      C[i + j * n] = sum / L[i + i * n];
    }
  }
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      real_type sum = C[i + j * n];
      for (int k = 0; k < j; ++k) {
        sum -= L[j + k * n] * A[i + k * n];
      }
      A[i + j * n] = sum / L[j + j * n];
    }
  }
  
  cuda_jacobi_eigen_host(n, A, w, eigvecs);
  
  for (int col = 0; col < n; ++col) {
    for (int i = n - 1; i >= 0; --i) {
      real_type sum = eigvecs[i + col * n];
      for (int k = i + 1; k < n; ++k) {
        sum -= L[k + i * n] * C[k + col * n];
      }
      C[i + col * n] = sum / L[i + i * n];
    }
    for (int i = 0; i < n; ++i) {
      eigvecs[i + col * n] = C[i + col * n];
    }
  }
  
  free(L);
  free(C);
}

