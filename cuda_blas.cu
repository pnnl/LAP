#include "cublas_v2.h"

#include <cusparse.h> 
#include "cuda_blas.h"
#if USE_FP64
#define cuda_data_type CUDA_R_64F
#else
#define cuda_data_type CUDA_R_32F
#endif

// Error checking macros
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
              __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define CUBLAS_CHECK(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
              __FILE__, __LINE__, status); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define CUSPARSE_CHECK(call) \
  do { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
      fprintf(stderr, "cuSPARSE error in %s at line %d: %d\n", \
              __FILE__, __LINE__, status); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)
static cublasHandle_t handle_cublas;
static cusparseHandle_t handle_cusparse;

static void *mv_buffer;
static void *L_buffer;
static void *U_buffer;
static void *ichol_buffer; // in ichol, we can get away with one buffer

static cusparseSpMatDescr_t matA = NULL;
static cusparseSpMatDescr_t matL;
static cusparseSpMatDescr_t matU;
static cusparseMatDescr_t descrL, descrU, descrLt, descrM; // last two are used only for incomplete CHolesky
static  csrsv2Info_t infoL, infoU, infoLt;
csric02Info_t infoM  = 0; // used only for Incomplete Cholesky

#define policy CUSPARSE_SOLVE_POLICY_USE_LEVEL 


void initialize_handles(){
  //printf("initializing handles! \n");
  CUBLAS_CHECK(cublasCreate(&handle_cublas));
  CUSPARSE_CHECK(cusparseCreate(&handle_cusparse));
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

  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX,
                                     n,
                                     (void*) x,
                                     cuda_data_type));

  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY,
                                     n,
                                     (void *) result,
                                     cuda_data_type));

  CUSPARSE_CHECK(cusparseCreateCsr(&matA,
                                   n,
                                   n,
                                   nnz,
                                   ia,
                                   ja,
                                   a,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   cuda_data_type));

  CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle_cusparse,
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
                                         &mv_buffer_size));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMalloc(&mv_buffer, mv_buffer_size));

  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
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

  CUSPARSE_CHECK(cusparseCreateMatDescr(&(descrL)));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CHECK(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));

  CUSPARSE_CHECK(cusparseCreateMatDescr(&(descrU)));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CHECK(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
  CUSPARSE_CHECK(cusparseCreateCsrsv2Info(&infoL));
  CUSPARSE_CHECK(cusparseCreateCsrsv2Info(&infoU));
  int L_buffer_size;  
  int U_buffer_size;  
#if USE_FP64 // it is double 
    CUSPARSE_CHECK(cusparseDcsrsv2_bufferSize(handle_cusparse, 
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                              n, 
                                              nnzL, 
                                              descrL,
                                              la, 
                                              lia, 
                                              lja,
                                              infoL, 
                                              &L_buffer_size));
    CUDA_CHECK(cudaMalloc((void**)&(L_buffer), L_buffer_size));

    CUSPARSE_CHECK(cusparseDcsrsv2_bufferSize(handle_cusparse, 
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                              n, 
                                              nnzU, 
                                              descrU,
                                              ua, 
                                              uia, 
                                              uja,
                                              infoU, 
                                              &U_buffer_size));
    CUDA_CHECK(cudaMalloc((void**)&(U_buffer), U_buffer_size));

    CUSPARSE_CHECK(cusparseDcsrsv2_analysis(handle_cusparse, 
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            n,
                                            nnzL,
                                            descrL,
                                            la,
                                            lia,
                                            lja,
                                            infoL,
                                            policy, 
                                            L_buffer));

    CUSPARSE_CHECK(cusparseDcsrsv2_analysis(handle_cusparse, 
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            n,
                                            nnzU,
                                            descrU,
                                            ua,
                                            uia,
                                            uja,
                                            infoU,
                                            policy, 
                                            U_buffer));
#else    
// it is 4 bytes so SINGLE

    CUSPARSE_CHECK(cusparseScsrsv2_bufferSize(handle_cusparse, 
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                              n, 
                                              nnzL, 
                                              descrL,
                                              la, 
                                              lia, 
                                              lja,
                                              infoL, 
                                              &L_buffer_size));
    CUDA_CHECK(cudaMalloc((void**)&(L_buffer), L_buffer_size));

    CUSPARSE_CHECK(cusparseScsrsv2_bufferSize(handle_cusparse, 
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                              n, 
                                              nnzU, 
                                              descrU,
                                              ua, 
                                              uia, 
                                              uja,
                                              infoU, 
                                              &U_buffer_size));
    CUDA_CHECK(cudaMalloc((void**)&(U_buffer), U_buffer_size));

    CUSPARSE_CHECK(cusparseScsrsv2_analysis(handle_cusparse, 
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            n,
                                            nnzL,
                                            descrL,
                                            la,
                                            lia,
                                            lja,
                                            infoL,
                                            policy, 
                                            L_buffer));

    CUSPARSE_CHECK(cusparseScsrsv2_analysis(handle_cusparse, 
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            n,
                                            nnzU,
                                            descrU,
                                            ua,
                                            uia,
                                            uja,
                                            infoU,
                                            policy, 
                                            U_buffer));
#endif
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

    CUSPARSE_CHECK(cusparseCreateCsr(&matL,
                                     n,
                                     n,
                                     nnzL,
                                     lia,
                                     lja,
                                     la,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cuda_data_type));

    CUSPARSE_CHECK(cusparseCreateCsr(&matU,
                                     n,
                                     n,
                                     nnzU,
                                     uia,
                                     uja,
                                     ua,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cuda_data_type));
}

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a)
{

  printf("initializing ICHOL \n");
  CUSPARSE_CHECK(cusparseCreateMatDescr(&descrM));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CHECK(cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL));

  CUSPARSE_CHECK(cusparseCreateMatDescr(&descrL));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CHECK(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CHECK(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
  CUSPARSE_CHECK(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT));

  CUSPARSE_CHECK(cusparseCreateCsric02Info(&infoM));
  CUSPARSE_CHECK(cusparseCreateCsrsv2Info(&infoL));
  CUSPARSE_CHECK(cusparseCreateCsrsv2Info(&infoLt));
  int structural_zero;
  int numerical_zero; 

  cusparseStatus_t status_cusparse;

  /* figure out the buffer size */

  int bufferSize, bufferSizeL, bufferSizeLt, bufferSizeM;
#if USE_FP64 // it is double 
      CUSPARSE_CHECK(cusparseDcsric02_bufferSize(handle_cusparse, 
                                                 n, 
                                                 nnzA,
                                                 descrM, 
                                                 a,
                                                 ia, 
                                                 ja, 
                                                 infoM, 
                                                 &bufferSizeM));

      CUSPARSE_CHECK(cusparseDcsrsv2_bufferSize(handle_cusparse, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n, 
                                                nnzA,
                                                descrL, 
                                                a, 
                                                ia, 
                                                ja, 
                                                infoL, 
                                                &bufferSizeL));

      CUSPARSE_CHECK(cusparseDcsrsv2_bufferSize(handle_cusparse, 
                                                CUSPARSE_OPERATION_TRANSPOSE, 
                                                n, 
                                                nnzA,
                                                descrL, 
                                                a, 
                                                ia, 
                                                ja, 
                                                infoLt, 
                                                &bufferSizeLt));
#else

      CUSPARSE_CHECK(cusparseScsric02_bufferSize(handle_cusparse, 
                                                 n, 
                                                 nnzA,
                                                 descrM, 
                                                 a,
                                                 ia, 
                                                 ja, 
                                                 infoM, 
                                                 &bufferSizeM));

      CUSPARSE_CHECK(cusparseScsrsv2_bufferSize(handle_cusparse, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n, 
                                                nnzA,
                                                descrL, 
                                                a, 
                                                ia, 
                                                ja, 
                                                infoL, 
                                                &bufferSizeL));

      CUSPARSE_CHECK(cusparseScsrsv2_bufferSize(handle_cusparse, 
                                                CUSPARSE_OPERATION_TRANSPOSE, 
                                                n, 
                                                nnzA,
                                                descrL, 
                                                a, 
                                                ia, 
                                                ja, 
                                                infoLt, 
                                                &bufferSizeLt));
#endif

  bufferSize = max(bufferSizeM, max(bufferSizeL, bufferSizeLt));

  CUDA_CHECK(cudaMalloc((void**) &ichol_buffer, bufferSize));

  /* and now analyze */

#if USE_FP64 // it is double 
  CUSPARSE_CHECK(cusparseDcsric02_analysis(handle_cusparse,
                                           n, 
                                           nnzA, 
                                           descrM,
                                           a, 
                                           ia, 
                                           ja, 
                                           infoM,
                                           policy, 
                                           ichol_buffer));
  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, infoM, &structural_zero);

  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have a problem: A(%d,%d) is missing\n", structural_zero, structural_zero);
  }

  /* analyze the solves as well */

  CUSPARSE_CHECK(cusparseDcsrsv2_analysis(handle_cusparse, 
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                          n, 
                                          nnzA, 
                                          descrL,
                                          a, 
                                          ia, 
                                          ja,
                                          infoL, 
                                          policy, 
                                          ichol_buffer));

  CUSPARSE_CHECK(cusparseDcsrsv2_analysis(handle_cusparse, 
                                          CUSPARSE_OPERATION_TRANSPOSE, 
                                          n, 
                                          nnzA, 
                                          descrL,
                                          a, 
                                          ia, 
                                          ja,
                                          infoLt, 
                                          policy, 
                                          ichol_buffer));

  /* decompose */
  CUSPARSE_CHECK(cusparseDcsric02(handle_cusparse, 
                                  n, 
                                  nnzA, 
                                  descrM,
                                  a, 
                                  ia, 
                                  ja, 
                                  infoM, 
                                  policy, 
                                  ichol_buffer));

  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, 
                                               infoM, 
                                               &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have another problem: L(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }
#else

  CUSPARSE_CHECK(cusparseScsric02_analysis(handle_cusparse,
                                           n, 
                                           nnzA, 
                                           descrM,
                                           a, 
                                           ia, 
                                           ja, 
                                           infoM,
                                           policy, 
                                           ichol_buffer));
  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, infoM, &structural_zero);

  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have a problem: A(%d,%d) is missing\n", structural_zero, structural_zero);
  }

  /* analyze the solves as well */

  CUSPARSE_CHECK(cusparseScsrsv2_analysis(handle_cusparse, 
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                          n, 
                                          nnzA, 
                                          descrL,
                                          a, 
                                          ia, 
                                          ja,
                                          infoL, 
                                          policy, 
                                          ichol_buffer));

  CUSPARSE_CHECK(cusparseScsrsv2_analysis(handle_cusparse, 
                                          CUSPARSE_OPERATION_TRANSPOSE, 
                                          n, 
                                          nnzA, 
                                          descrL,
                                          a, 
                                          ia, 
                                          ja,
                                          infoLt, 
                                          policy, 
                                          ichol_buffer));

  /* decompose */
  CUSPARSE_CHECK(cusparseScsric02(handle_cusparse, 
                                  n, 
                                  nnzA, 
                                  descrM,
                                  a, 
                                  ia, 
                                  ja, 
                                  infoM, 
                                  policy, 
                                  ichol_buffer));

  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, 
                                               infoM, 
                                               &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have another problem: L(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }
#endif
}


void cuda_ichol(const int *ia, 
                const int *ja, 
                real_type *a, 
                const int nnzA,
                pdata *prec_data, 
                real_type *x, 
                real_type *y) {
  real_type one = 1.0;

#if USE_FP64 // it is double 
  CUSPARSE_CHECK(cusparseDcsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                       prec_data->n, 
                                       nnzA, 
                                       &one, 
                                       descrL, // replace with cusparseSpSV
                                       prec_data->ichol_vals, 
                                       ia, 
                                       ja, 
                                       infoL,
                                       x,//input 
                                       prec_data->aux_vec1, //output
                                       policy, 
                                       ichol_buffer));

  /* solve L'*y = aux_vec1 */
  CUSPARSE_CHECK(cusparseDcsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_TRANSPOSE, 
                                       prec_data->n, 
                                       nnzA, &one, 
                                       descrL, // replace with cusparseSpSV
                                       prec_data->ichol_vals, 
                                       ia, 
                                       ja, 
                                       infoLt,
                                       prec_data->aux_vec1, 
                                       y, 
                                       policy, 
                                       ichol_buffer));
#else //sp

  CUSPARSE_CHECK(cusparseScsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                       prec_data->n, 
                                       nnzA, 
                                       &one, 
                                       descrL, // replace with cusparseSpSV
                                       prec_data->ichol_vals, 
                                       ia, 
                                       ja, 
                                       infoL,
                                       x,//input 
                                       prec_data->aux_vec1, //output
                                       policy, 
                                       ichol_buffer));

  /* solve L'*y = aux_vec1 */
  CUSPARSE_CHECK(cusparseScsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_TRANSPOSE, 
                                       prec_data->n, 
                                       nnzA, &one, 
                                       descrL, // replace with cusparseSpSV
                                       prec_data->ichol_vals, 
                                       ia, 
                                       ja, 
                                       infoLt,
                                       prec_data->aux_vec1, 
                                       y, 
                                       policy, 
                                       ichol_buffer));
#endif
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

#if USE_FP64
  CUBLAS_CHECK(cublasDdot(handle_cublas, 
                          n, 
                          v, 
                          1, 
                          w, 
                          1, 
                          &sum));
#else
  CUBLAS_CHECK(cublasSdot(handle_cublas, 
                          n, 
                          v, 
                          1, 
                          w, 
                          1, 
                          &sum));
#endif
  return sum;
}

void cuda_scal (const int n, const real_type alpha, real_type *v){
#if USE_FP64 
  CUBLAS_CHECK(cublasDscal(handle_cublas, 
                           n,
                           &alpha,
                           v, 
                           1));
#else
  CUBLAS_CHECK(cublasSscal(handle_cublas, 
                           n,
                           &alpha,
                           v, 
                           1));
#endif
}

void cuda_axpy (const int n, const real_type alpha, const real_type *x, real_type *y){

#if USE_FP64 
  CUBLAS_CHECK(cublasDaxpy(handle_cublas, 
                           n,
                           &alpha,
                           x, 
                           1,
                           y, 
                           1));
#else
  CUBLAS_CHECK(cublasSaxpy(handle_cublas, 
                           n,
                           &alpha,
                           x, 
                           1,
                           y, 
                           1));
#endif
}

void cuda_csr_matvec(const int n, const int nnz, const int *ia, const int *ja, const real_type *a, const real_type *x, real_type *result, const real_type*al, const real_type *bet){
  /* y = alpha *A* x + beta * y */ 

  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  cusparseSpMatDescr_t matCSR;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX,
                                     n,
                                     (void*) x,
                                     cuda_data_type));

  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY,
                                     n,
                                     (void *) result,
                                     cuda_data_type));

  CUSPARSE_CHECK(cusparseCreateCsr(&matCSR,
                                   n,
                                   n,
                                   nnz,
                                   (void *)ia,
                                   (void *)ja,
                                   (void *)a,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   cuda_data_type));

  CUSPARSE_CHECK(cusparseSpMV(handle_cusparse,
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
                              mv_buffer));

  CUSPARSE_CHECK(cusparseDestroySpMat(matCSR));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
}

void cuda_lower_triangular_solve(const int n,
                                 const int nnzL, 
                                 const int *lia, 
                                 const int *lja, 
                                 const real_type *la,
                                 const real_type *diagonal, 
                                 const real_type *x, real_type *result){
  /* compute result = L^{-1}x */
  /* we DO NOT assume anything about L diagonal */
  /* d_x3 = L^(-1)dx2 */

  real_type one = 1.0;
#if USE_FP64
  CUSPARSE_CHECK(cusparseDcsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
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
                                       policy,
                                       L_buffer));
#else
  CUSPARSE_CHECK(cusparseScsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
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
                                       policy,
                                       L_buffer));
#endif
}


void cuda_upper_triangular_solve(const int n, 
                                 const int nnzU, 
                                 const int *uia, 
                                 const int *uja, 
                                 const real_type *ua, 
                                 const real_type *diagonal, 
                                 const real_type *x, 
                                 real_type *result){

  /* compute result = U^{-1}x */
  real_type one = 1.0;
#if USE_FP64 
  CUSPARSE_CHECK(cusparseDcsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
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
                                       policy,
                                       U_buffer));
#else
  CUSPARSE_CHECK(cusparseScsrsv2_solve(handle_cusparse, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
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
                                       policy,
                                       U_buffer));
#endif
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

  CUDA_CHECK(cudaMemcpy(dest, src, sizeof(real_type) * n, cudaMemcpyDeviceToDevice));
}

void cuda_vec_zero(const int n, real_type *vec){

  cuda_vec_zero_kernel<<<1024, 1024>>>(n, vec);
}

