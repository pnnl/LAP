#include <rocsparse.h>
#include <rocblas.h>
#include <hip/hip_runtime_api.h>
#include "hip_blas.h"


static rocblas_handle handle_rocblas;
static  rocsparse_handle  handle_rocsparse;
static void *mv_buffer = NULL;
static void *L_buffer;
static void *U_buffer;
static void *ichol_buffer;

static rocsparse_mat_descr matA = NULL;
static rocsparse_mat_descr descrL, descrU, descrA;
static rocsparse_mat_descr descrLt; //for ICHOL
static rocsparse_mat_info  infoL, infoU, infoLic, infoLtic;
static rocsparse_mat_info  infoA;
static rocsparse_mat_descr descrLic, descrLtic, descrM; //ICHOL
static rocsparse_mat_info infoM; //ICHOL


void initialize_handles(){

  rocblas_create_handle(&handle_rocblas);
  rocsparse_create_handle(&handle_rocsparse);

  rocsparse_create_mat_descr(&(descrL));
  rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower);
  rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero);

  rocsparse_create_mat_descr(&(descrU));
  rocsparse_set_mat_index_base(descrU, rocsparse_index_base_zero);
  rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper);

  rocsparse_create_mat_descr(&(descrA));
  rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero);
  rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general);

  rocsparse_create_mat_info(&infoA);
  rocsparse_create_mat_info(&infoL);
  rocsparse_create_mat_info(&infoU);
  hipDeviceSynchronize();
}

void analyze_spmv(const int n, 
                  const int nnz, 
                  int *ia, 
                  int *ja, 
                  real_type *a, 
                  const real_type *x, 
                  real_type *result,
                  char * option
                 ){
  /* no buffer in matvec */
  rocsparse_status status_rocsparse;
  if (strcmp(option, "A") == 0) {
    status_rocsparse = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                                 rocsparse_operation_none,
                                                 n,
                                                 n,
                                                 nnz,
                                                 descrA,
                                                 a,
                                                 ia,
                                                 ja,
                                                 infoA);
  }

  if (strcmp(option, "L") == 0) {
    status_rocsparse = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                                 rocsparse_operation_none,
                                                 n,
                                                 n,
                                                 nnz,
                                                 descrL,
                                                 a,
                                                 ia,
                                                 ja,
                                                 infoL);
  }

  if (strcmp(option, "U") == 0) {
    status_rocsparse = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                                 rocsparse_operation_none,
                                                 n,
                                                 n,
                                                 nnz,
                                                 descrU,
                                                 a,
                                                 ia,
                                                 ja,
                                                 infoU);
  }

  if (status_rocsparse != 0) {
    printf("mv analysis status for %s is %d \n", option, status_rocsparse);
  }
  hipDeviceSynchronize();
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

  size_t L_buffer_size;  
  size_t U_buffer_size;  
  rocsparse_status status_rocsparse;

  status_rocsparse = rocsparse_dcsrsv_buffer_size(handle_rocsparse, 
                                                  rocsparse_operation_none, 
                                                  n, 
                                                  nnzL, 
                                                  descrL,
                                                  la, 
                                                  lia, 
                                                  lja,
                                                  infoL, 
                                                  &L_buffer_size);

  hipMalloc((void**)&(L_buffer), L_buffer_size);

  status_rocsparse = rocsparse_dcsrsv_buffer_size(handle_rocsparse, 
                                                  rocsparse_operation_none, 
                                                  n, 
                                                  nnzU, 
                                                  descrU,
                                                  ua, 
                                                  uia, 
                                                  uja,
                                                  infoU, 
                                                  &U_buffer_size);
  hipMalloc((void**)&(U_buffer), U_buffer_size);

  status_rocsparse = rocsparse_dcsrsv_analysis(handle_rocsparse, 
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
                                               L_buffer);
  if (status_rocsparse != 0) {
    printf("status after analysis 1 %d \n", status_rocsparse);
  }

  status_rocsparse = rocsparse_dcsrsv_analysis(handle_rocsparse, 
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
                                               U_buffer);
  if (status_rocsparse != 0) {
    printf("status after analysis 2 %d \n", status_rocsparse);
  }
  hipDeviceSynchronize();
}

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a)
{
  // printf("initializing ICHOLi, n = %d, nnzA = %d \n",n,nnzA);
  /* Create matrix descriptor for M */
  rocsparse_status status_rocsparse;
  rocsparse_create_mat_descr(&descrM);
  rocsparse_set_mat_type(descrM, rocsparse_matrix_type_general);

  /* Create matrix descriptor for L */
  rocsparse_create_mat_descr(&descrLic);
  rocsparse_set_mat_fill_mode(descrLic, rocsparse_fill_mode_lower);
  rocsparse_set_mat_diag_type(descrLic, rocsparse_diag_type_non_unit);
  rocsparse_set_mat_index_base(descrLic, rocsparse_index_base_zero);

  /* Create matrix descriptor for L' */
  rocsparse_create_mat_descr(&descrLtic);
  rocsparse_set_mat_fill_mode(descrLtic, rocsparse_fill_mode_upper);
  rocsparse_set_mat_diag_type(descrLtic, rocsparse_diag_type_non_unit);
  rocsparse_set_mat_index_base(descrLtic, rocsparse_index_base_zero);

  /* Create matrix info structure */
  rocsparse_create_mat_info(&infoM);
  rocsparse_create_mat_info(&infoLic);
  rocsparse_create_mat_info(&infoLtic);

  /* Obtain required buffer size */
  size_t buffer_size_M;
  size_t buffer_size_L;
  size_t buffer_size_Lt;

  rocsparse_dcsric0_buffer_size(handle_rocsparse,
                                n,
                                nnzA,
                                descrM,
                                a,
                                ia,
                                ja,
                                infoM,
                                &buffer_size_M);

  rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                               rocsparse_operation_none,
                               n,
                               nnzA,
                               descrLic,
                               a,
                               ia,
                               ja,
                               infoM,
                               &buffer_size_L);

  rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                               rocsparse_operation_transpose,
                               n,
                               nnzA,
                               descrLic,
                               a,
                               ia,
                               ja,
                               infoM,
                               &buffer_size_Lt);
  // printf("Buffer sizes: %d %d %d \n", buffer_size_M, buffer_size_L, buffer_size_Lt);
  size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));
  // printf("finalsize %d \n",buffer_size);
  // Allocate temporary buffer
  hipMalloc(&ichol_buffer, buffer_size);

  /* Perform analysis steps, using rocsparse_analysis_policy_reuse to improve 
   * computation performance */
  status_rocsparse = rocsparse_dcsric0_analysis(handle_rocsparse,
                                                n,
                                                nnzA,
                                                descrM,
                                                a,
                                                ia,
                                                ja,
                                                infoM,
                                                rocsparse_analysis_policy_reuse,
                                                rocsparse_solve_policy_auto,
                                                ichol_buffer);
  // printf("status 0: %d \n", status_rocsparse);
  status_rocsparse = rocsparse_dcsrsv_analysis(handle_rocsparse,
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
                                               ichol_buffer);
  // printf("status 1: %d \n", status_rocsparse);
  status_rocsparse = rocsparse_dcsrsv_analysis(handle_rocsparse,
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
                                               ichol_buffer);
  // printf("status 2: %d \n", status_rocsparse);

  /* Check for zero pivot */
  rocsparse_int position;
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
                                                                 infoM,
                                                                 &position)) {
    printf("A has structural zero at A(%d,%d)\n", position, position);
  }

  /* Compute incomplete Cholesky factorization M = LL' */
  status_rocsparse =	rocsparse_dcsric0(handle_rocsparse,
                                        n,
                                        nnzA,
                                        descrM,
                                        a,
                                        ia,
                                        ja,
                                        infoM,
                                        rocsparse_solve_policy_auto,
                                        ichol_buffer);
  // printf("status 3: %d \n", status_rocsparse);

  /* Check for zero pivot */
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
                                                                 infoM,
                                                                 &position)) {
    printf("L has structural and/or numerical zero at L(%d,%d)\n",
           position,
           position);
  }
    hipDeviceSynchronize();
}

void hip_ichol(const int *ia, 
               const int *ja, 
               real_type *a, 
               const int nnzA, 
               pdata *prec_data, 
               real_type *x, 
               real_type *y)
{
  real_type one = 1.0;
  rocsparse_status st;
  st = rocsparse_dcsrsv_solve(handle_rocsparse,
                              rocsparse_operation_none,
                              prec_data->n,
                              nnzA,
                              &one,
                              descrLic,
                              prec_data->ichol_vals,
                              ia,
                              ja,
                              infoM,
                              x,//input
                              prec_data->aux_vec1, //output 
                              rocsparse_solve_policy_auto,
                              ichol_buffer);

  if (st != 0) {
    printf("before L^T solve: norm of input %16.16e, norm of output %16.16e\n", hip_dot (prec_data->n, prec_data->aux_vec1, prec_data->aux_vec1), hip_dot (prec_data->n, y, y) );
  } 
  /* Solve L'y = z */
  st = rocsparse_dcsrsv_solve(handle_rocsparse,
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
                              ichol_buffer);
  if (st != 0) {
    printf("status L^T solve: %d \n", st);
  }
  hipDeviceSynchronize();
}

__global__ void hip_vec_vec_kernel(const int n,
                                   const real_type *x,
                                   const real_type *y,
                                   real_type *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n) {
    z[idx] = x[idx] * y[idx];

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_reciprocal_kernel(const int n,
                                          const real_type *x,
                                          real_type *z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n) {
    if  (x[idx] != 0.0){ 
      z[idx] = 1.0 / x[idx];
    } else { 
      z[idx] = 0.0;
    }

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_sqrt_kernel(const int n,
                                    const real_type *x,
                                    real_type *z){
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
                                   real_type *x){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    x[idx] = value;

    idx += blockDim.x * gridDim.x;
  }
}

__global__ void hip_vec_zero_kernel(const int n,
                                    real_type *x){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    x[idx] =  0.0;

    idx += blockDim.x * gridDim.x;
  }
}

real_type hip_dot (const int n, const real_type *v, const real_type *w){
  real_type sum;

  hipDeviceSynchronize();
  rocblas_ddot (handle_rocblas, 
                n, 
                v, 
                1, 
                w, 
                1, 
                &sum);
  hipDeviceSynchronize();
  return sum;
}

void hip_scal (const int n, const real_type alpha, real_type *v){
  rocblas_dscal(handle_rocblas, 
                n,
                &alpha,
                v, 
                1);
  hipDeviceSynchronize();
}

void hip_axpy (const int n, const real_type alpha, const real_type *x, real_type *y){
  rocblas_daxpy(handle_rocblas, 
                n,
                &alpha,
                x, 
                1,
                y, 
                1);
  hipDeviceSynchronize();
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
                    const char *kind){
  /* y = alpha *A* x + beta *y */
  rocsparse_status st;
  hipDeviceSynchronize();
  if (strcmp(kind, "A") == 0) {
    st = rocsparse_dcsrmv(handle_rocsparse,
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
                          result);
  } 

  if (strcmp(kind, "L") == 0) {
    st = rocsparse_dcsrmv(handle_rocsparse,
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
                          result);
  }

  if (strcmp(kind, "U") == 0) {
    st = rocsparse_dcsrmv(handle_rocsparse,
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
                          result);
  }
  hipDeviceSynchronize();
  // printf("status after mv: %d\n", st);
}

void hip_lower_triangular_solve(const int n,
                                const int nnzL, 
                                const int *lia, 
                                const int *lja, 
                                const real_type *la,
                                const real_type *diagonal, 
                                const real_type *x, 
                                real_type *result){
  /* compute result = L^{-1}x */
  /* we DO NOT assume anything about L diagonal */
  /* d_x3 = L^(-1)dx2 */
  real_type one = 1.0;

  hipDeviceSynchronize();
  rocsparse_dcsrsv_solve(handle_rocsparse, 
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
                         L_buffer);
  hipDeviceSynchronize();
}

void hip_upper_triangular_solve(const int n, 
                                const int nnzU, 
                                const int *uia, 
                                const int *uja, 
                                const real_type *ua, 
                                const real_type *diagonal, 
                                const real_type *x, 
                                real_type *result){
  /* compute result = U^{-1}x */
  real_type one = 1.0;
  hipDeviceSynchronize();
  rocsparse_dcsrsv_solve(handle_rocsparse, 
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
                         U_buffer);
  hipDeviceSynchronize();
}

/* not std blas but needed and embarassingly parallel */ 

/* hip vec-vec computes an element-wise product (needed for scaling) */

void hip_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res){
  hipLaunchKernelGGL(hip_vec_vec_kernel, dim3(n / 1024 + 1), dim3(1024), 0, 0, n, x, y, res);
  hipDeviceSynchronize();
}

/*vector reciprocal computes 1./d */ 

void hip_vector_reciprocal(const int n, const real_type *v, real_type *res){
  hipLaunchKernelGGL( hip_vec_reciprocal_kernel,dim3(n / 1024 + 1), dim3(1024), 0, 0, n, v, res);
  hipDeviceSynchronize();
}

//vector sqrt takes an sqrt from each vector entry 

void hip_vector_sqrt(const int n, const real_type *v, real_type *res){
  hipLaunchKernelGGL(hip_vec_sqrt_kernel, dim3(n / 1024 +1), dim3(1024), 0,0,n, v, res);
  hipDeviceSynchronize();
}

void hip_vec_copy(const int n, const real_type *src, real_type *dest){
  hipMemcpy(dest, src, sizeof(real_type) * n, hipMemcpyDeviceToDevice);
  hipDeviceSynchronize();
}


void hip_vec_set(const int n, real_type value, real_type *vec){
  hipLaunchKernelGGL(hip_vec_set_kernel,dim3(n/ 1024 + 1), dim3(1024), 0, 0, n, value, vec);
  hipDeviceSynchronize();
}

void hip_vec_zero(const int n, real_type *vec){
  hipLaunchKernelGGL(hip_vec_zero_kernel,dim3(n / 1024 + 1), dim3(1024), 0, 0, n, vec);
  hipDeviceSynchronize();
}
