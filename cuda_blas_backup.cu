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
static void *ichol_buffer; // in ichol, we can get away with one buffer (?)
static void *ichol_Lbuffer;
static void *ichol_Ltbuffer;


static cusparseSpMatDescr_t matA = NULL;
static cusparseSpMatDescr_t matL;
static cusparseSpMatDescr_t matU;
static cusparseMatDescr_t descrLt, descrM; // last two are used only for incomplete CHolesky
#if (H100 != 1)
static  csrsv2Info_t infoL, infoU, infoLt;
static cusparseMatDescr_t descrL, descrU;
#else
static cusparseSpSVDescr_t  descrL, descrU;
static cusparseSpSVDescr_t descrLichol;
static cusparseSpSVDescr_t descrLtichol;
#endif
csric02Info_t infoM  = 0; // used only for Incomplete Cholesky

#define policy CUSPARSE_SOLVE_POLICY_USE_LEVEL 


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
					  real_type *ua,
					  real_type *vecx_data,
					  real_type *vecy_data)
{

#if (H100 != 1)
  int L_buffer_size;  
  int U_buffer_size; 
  cusparseCreateMatDescr(&(descrL));
  cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);

  cusparseCreateMatDescr(&(descrU));
  cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
  cusparseCreateCsrsv2Info(&infoL);
  cusparseCreateCsrsv2Info(&infoU);
#if USE_FP64 // it is double 
  cusparseDcsrsv2_bufferSize(handle_cusparse, 
			     CUSPARSE_OPERATION_NON_TRANSPOSE, 
			     n, 
			     nnzL, 
			     descrL,
			     la, 
			     lia, 
			     lja,
			     infoL, 
			     &L_buffer_size);
  //printf("buffer size L %d\n", L_buffer_size);
  cudaMalloc((void**)&(L_buffer), L_buffer_size);

  cusparseDcsrsv2_bufferSize(handle_cusparse, 
			     CUSPARSE_OPERATION_NON_TRANSPOSE, 
			     n, 
			     nnzU, 
			     descrU,
			     ua, 
			     uia, 
			     uja,
			     infoU, 
			     &U_buffer_size);
  //printf("buffer size U %d\n", U_buffer_size);
  cudaMalloc((void**)&(U_buffer), U_buffer_size);
  cusparseStatus_t status_cusparse;
  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE,
					     n,
					     nnzL,
					     descrL,
					     la,
					     lia,
					     lja,
					     infoL,
					     policy, 
					     L_buffer);

  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE,
					     n,
					     nnzU,
					     descrU,
					     ua,
					     uia,
					     uja,
					     infoU,
					     policy, 
					     U_buffer);
#else    
  // it is 4 bytes so SINGLE

  cusparseScsrsv2_bufferSize(handle_cusparse, 
			     CUSPARSE_OPERATION_NON_TRANSPOSE, 
			     n, 
			     nnzL, 
			     descrL,
			     la, 
			     lia, 
			     lja,
			     infoL, 
			     &L_buffer_size);
  //printf("buffer size L %d\n", L_buffer_size);
  cudaMalloc((void**)&(L_buffer), L_buffer_size);

  cusparseScsrsv2_bufferSize(handle_cusparse, 
			     CUSPARSE_OPERATION_NON_TRANSPOSE, 
			     n, 
			     nnzU, 
			     descrU,
			     ua, 
			     uia, 
			     uja,
			     infoU, 
			     &U_buffer_size);
  //printf("buffer size U %d\n", U_buffer_size);
  cudaMalloc((void**)&(U_buffer), U_buffer_size);
  cusparseStatus_t status_cusparse;
  status_cusparse = cusparseScsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE,
					     n,
					     nnzL,
					     descrL,
					     la,
					     lia,
					     lja,
					     infoL,
					     policy, 
					     L_buffer);

  status_cusparse = cusparseScsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE,
					     n,
					     nnzU,
					     descrU,
					     ua,
					     uia,
					     uja,
					     infoU,
					     policy, 
					     U_buffer);
#endif
  //end of FP32 code
#else //We ARE using cuda/12 or H100

  size_t L_buffer_size;  
  size_t U_buffer_size; 
#if USE_FP64 // it is double 
  cusparseCreateCsr(&matL, 
		    n, 
		    n, 
		    nnzL,
		    lia, 
		    lja, 
		    la,
		    CUSPARSE_INDEX_32I, 
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_BASE_ZERO, 
		    CUDA_R_64F);
  cusparseCreateCsr(&matU, 
		    n, 
		    n, 
		    nnzU,
		    uia, 
		    uja, 
		    ua,
		    CUSPARSE_INDEX_32I, 
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_BASE_ZERO, 
		    CUDA_R_64F);
  //descriptors
  cusparseSpSV_createDescr(&descrL);
  cusparseSpSV_createDescr(&descrU);

  //set fill modes
  cusparseFillMode_t fillmodeL = CUSPARSE_FILL_MODE_LOWER;
  cusparseFillMode_t fillmodeU = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_FILL_MODE,
			    &fillmodeL, 
			    sizeof(fillmodeL)); 
  cusparseSpMatSetAttribute(matU, 
			    CUSPARSE_SPMAT_FILL_MODE,
			    &fillmodeU, 
			    sizeof(fillmodeU)); 

  //set diag types (note: both are NON UNIT)
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_DIAG_TYPE,
			    &diagtype, 
			    sizeof(diagtype));
  cusparseSpMatSetAttribute(matU, 
			    CUSPARSE_SPMAT_DIAG_TYPE,
			    &diagtype, 
			    sizeof(diagtype));
  // vectors 
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;

  cusparseCreateDnVec(&vecX, n, vecx_data, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, n, vecy_data, CUDA_R_64F);

  // allocate buffers
  real_type alpha = 1.0;
  cusparseSpSV_bufferSize(handle_cusparse, 
			  CUSPARSE_OPERATION_NON_TRANSPOSE,
			  &alpha,
			  matL, 
			  vecX, 
			  vecY, 
			  CUDA_R_64F,
			  CUSPARSE_SPSV_ALG_DEFAULT, 
			  descrL,
			  &L_buffer_size);
  cudaMalloc(&L_buffer, L_buffer_size);
  cusparseSpSV_bufferSize(handle_cusparse, 
			  CUSPARSE_OPERATION_NON_TRANSPOSE,
			  &alpha,
			  matU, 
			  vecX, 
			  vecY, 
			  CUDA_R_64F,
			  CUSPARSE_SPSV_ALG_DEFAULT, 
			  descrU,
			  &U_buffer_size);
  cudaMalloc(&U_buffer, L_buffer_size);
  // analyze
  cusparseSpSV_analysis(handle_cusparse, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matL, 
			vecX, 
			vecY, 
			CUDA_R_64F,
			CUSPARSE_SPSV_ALG_DEFAULT, 
			descrL,
			L_buffer);

  cusparseSpSV_analysis(handle_cusparse, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matU, 
			vecX, 
			vecY, 
			CUDA_R_64F,
			CUSPARSE_SPSV_ALG_DEFAULT, 
			descrU,
			U_buffer);

#else // same code but for single

  cusparseCreateCsr(&matL, 
		    n, 
		    n, 
		    nnzL,
		    lia, 
		    lja, 
		    la,
		    CUSPARSE_INDEX_32I, 
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_BASE_ZERO, 
		    CUDA_R_32F);
  cusparseCreateCsr(&matU, 
		    n, 
		    n, 
		    nnzU,
		    uia, 
		    uja, 
		    ua,
		    CUSPARSE_INDEX_32I, 
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_BASE_ZERO, 
		    CUDA_R_32F);
  //descriptors
  cusparseSpSV_createDescr(&descrL);
  cusparseSpSV_createDescr(&descrU);

  //set fill modes
  cusparseFillMode_t fillmodeL = CUSPARSE_FILL_MODE_LOWER;
  cusparseFillMode_t fillmodeU = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_FILL_MODE,
			    &fillmodeL, 
			    sizeof(fillmodeL)); 
  cusparseSpMatSetAttribute(matU, 
			    CUSPARSE_SPMAT_FILL_MODE,
			    &fillmodeU, 
			    sizeof(fillmodeU)); 

  //set diag types (note: both are NON UNIT)
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_DIAG_TYPE,
			    &diagtype, 
			    sizeof(diagtype));
  cusparseSpMatSetAttribute(matU, 
			    CUSPARSE_SPMAT_DIAG_TYPE,
			    &diagtype, 
			    sizeof(diagtype));
  // vectors 
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;

  cusparseCreateDnVec(&vecX, n, vecx_data, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, n, vecy_data, CUDA_R_32F);

  // allocate buffers
  real_type alpha = 1.0;
  cusparseSpSV_bufferSize(handle_cusparse, 
			  CUSPARSE_OPERATION_NON_TRANSPOSE,
			  &alpha,
			  matL, 
			  vecX, 
			  vecY, 
			  CUDA_R_32F,
			  CUSPARSE_SPSV_ALG_DEFAULT, 
			  descrL,
			  &L_buffer_size);
  cudaMalloc(&L_buffer, L_buffer_size);
  cusparseSpSV_bufferSize(handle_cusparse, 
			  CUSPARSE_OPERATION_NON_TRANSPOSE,
			  &alpha,
			  matU, 
			  vecX, 
			  vecY, 
			  CUDA_R_32F,
			  CUSPARSE_SPSV_ALG_DEFAULT, 
			  descrU,
			  &U_buffer_size);
  cudaMalloc(&U_buffer, L_buffer_size);
  // analyze
  cusparseSpSV_analysis(handle_cusparse, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matL, 
			vecX, 
			vecY, 
			CUDA_R_32F,
			CUSPARSE_SPSV_ALG_DEFAULT, 
			descrL,
			L_buffer);

  cusparseSpSV_analysis(handle_cusparse, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matU, 
			vecX, 
			vecY, 
			CUDA_R_32F,
			CUSPARSE_SPSV_ALG_DEFAULT, 
			descrU,
			U_buffer);
#endif // F32 vs F64
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
#endif //H100

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

  cusparseCreateCsr(&matL,
		    n,
		    n,
		    nnzL,
		    lia,
		    lja,
		    la,
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_BASE_ZERO,
		    cuda_data_type);

  cusparseCreateCsr(&matU,
		    n,
		    n,
		    nnzU,
		    uia,
		    uja,
		    ua,
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_32I,
		    CUSPARSE_INDEX_BASE_ZERO,
		    cuda_data_type);
}

void initialize_ichol(const int n, 
		      const int nnzA, 
		      int *ia, 
		      int *ja, 
		      real_type *a,
		      real_type *xdata,
		      real_type *ydata)
{
  cusparseCreateMatDescr(&descrM);
  cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseCreateCsric02Info(&infoM);

#if (H100 != 1)
  
  cusparseCreateMatDescr(&descrL);
  cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);

  cusparseCreateCsrsv2Info(&infoL);
  cusparseCreateCsrsv2Info(&infoLt);

#endif  

  int structural_zero;
  int numerical_zero; 

  cusparseStatus_t status_cusparse;

  /* figure out the buffer size */

  int bufferSize, bufferSizeL, bufferSizeLt, bufferSizeM;
#if USE_FP64 // it is double 
  status_cusparse =  cusparseDcsric02_bufferSize(handle_cusparse, 
						 n, 
						 nnzA,
						 descrM, 
						 a,
						 ia, 
						 ja, 
						 infoM, 
						 &bufferSizeM);

  if (status_cusparse != 0 ) printf("We have a problem (-2), status: %d \n", status_cusparse ); 
#if (H100 != 1)
  
  status_cusparse =  cusparseDcsrsv2_bufferSize(handle_cusparse, 
						CUSPARSE_OPERATION_NON_TRANSPOSE, 
						n, 
						nnzA,
						descrL, 
						a, 
						ia, 
						ja, 
						infoL, 
						&bufferSizeL);

  status_cusparse =  cusparseDcsrsv2_bufferSize(handle_cusparse, 
						CUSPARSE_OPERATION_TRANSPOSE, 
						n, 
						nnzA,
						descrL, 
						a, 
						ia, 
						ja, 
						infoLt, 
						&bufferSizeLt);
  bufferSize = max(bufferSizeM, max(bufferSizeL, bufferSizeLt));

  cudaMalloc((void**) &ichol_buffer, bufferSize);
#else
  // CUDA 12 aka H100
  cudaDataType dataType;
  if (sizeof(reaL_type) == 4) dataType = CUDA_R_32F;
  if (sizeof(reaL_type) == 8) dataType = CUDA_R_64F;
  size_t bL, bLt;
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;

  cusparseCreateDnVec(&vecX, n, xdata, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, n, ydata, CUDA_R_64F);
  double alpha = 1.0;
  status_cusparse = cusparseCreateCsr(&matL, 
				      n, 
				      n, 
				      nnzA,
				      ia, 
				      ja, 
				      a, 
				      CUSPARSE_INDEX_32I, 
				      CUSPARSE_INDEX_32I,
				      CUSPARSE_INDEX_BASE_ZERO, 
				      dataType);

  if (status_cusparse != 0 ) printf("We have a problem SV (0), status: %d \n", status_cusparse ); 
  
  cusparseSpSV_createDescr(&descrLichol);
  cusparseSpSV_createDescr(&descrLtichol);

  //set fill modes
  cusparseFillMode_t fillmodeL = CUSPARSE_FILL_MODE_LOWER;
  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_FILL_MODE,
			    &fillmodeL, 
			    sizeof(fillmodeL)); 

  //set diag types (note: both are NON UNIT)
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_DIAG_TYPE,
			    &diagtype, 
			    sizeof(diagtype));

  cusparseSpMatSetAttribute(matL, 
			    CUSPARSE_SPMAT_DIAG_TYPE,
			    &diagtype, 
			    sizeof(diagtype));

  status_cusparse = cusparseSpSV_bufferSize(handle_cusparse, 
					    CUSPARSE_OPERATION_NON_TRANSPOSE,
					    &alpha,
					    matL, 
					    NULL, 
					    NULL, 
					    dataType,
					    CUSPARSE_SPSV_ALG_DEFAULT, 
					    descrLichol,
					    &bL);
 
 status_cusparse = cusparseSpSV_bufferSize(handle_cusparse, 
					    CUSPARSE_OPERATION_TRANSPOSE,
					    &alpha,
					    matL, 
					    NULL, 
					    NULL, 
					    dataType,
					    CUSPARSE_SPSV_ALG_DEFAULT, 
					    descrLtichol,
					    &bLt);
  if (status_cusparse != 0 ) printf("We have a problem SV (1), status: %d \n", status_cusparse ); 
  bufferSizeL =  (int) bL;
  bufferSizeLt =  (int) bLt;

// we need separate buffers in this case
  cudaMalloc((void**) &ichol_buffer, bufferSizeM);
  cudaMalloc((void**) &ichol_Lbuffer, bufferSizeL);
  cudaMalloc((void**) &ichol_Ltbuffer, bufferSizeLt);
#endif
#else

  status_cusparse =  cusparseScsric02_bufferSize(handle_cusparse, 
						 n, 
						 nnzA,
						 descrM, 
						 a,
						 ia, 
						 ja, 
						 infoM, 
						 &bufferSizeM);

  status_cusparse =  cusparseScsrsv2_bufferSize(handle_cusparse, 
						CUSPARSE_OPERATION_NON_TRANSPOSE, 
						n, 
						nnzA,
						descrLichol, 
						a, 
						ia, 
						ja, 
						infoL, 
						&bufferSizeL);

  status_cusparse =  cusparseScsrsv2_bufferSize(handle_cusparse, 
						CUSPARSE_OPERATION_TRANSPOSE, 
						n, 
						nnzA,
						descrLichol, 
						a, 
						ia, 
						ja, 
						infoLt, 
						&bufferSizeLt);
  bufferSize = max(bufferSizeM, max(bufferSizeL, bufferSizeLt));

  cudaMalloc((void**) &ichol_buffer, bufferSize);
#endif
  /* and now analyze */

#if USE_FP64 // it is double 
  status_cusparse = cusparseDcsric02_analysis(handle_cusparse,
					      n, 
					      nnzA, 
					      descrM,
					      a, 
					      ia, 
					      ja, 
					      infoM,
					      policy, 
					      ichol_buffer);
  if (status_cusparse != 0 ) printf("We have a problem (-1), status: %d \n", status_cusparse ); 
  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, infoM, &structural_zero);

  cudaDeviceSynchronize();
  /* decompose */
  status_cusparse = cusparseDcsric02(handle_cusparse, 
				     n, 
				     nnzA, 
				     descrM,
				     a, 
				     ia, 
				     ja, 
				     infoM, 
				     policy, 
				     ichol_buffer);

  cudaDeviceSynchronize();
  if (status_cusparse != 0 ) printf("We have a problem (2), status: %d \n", status_cusparse ); 

  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have a problem: A(%d,%d) is missing\n", structural_zero, structural_zero);
  }



  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, 
					       infoM, 
					       &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have another problem: L(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }
  /* analyze the solves as well */

#if (H100 != 1)
  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE, 
					     n, 
					     nnzA, 
					     descrLichol,
					     a, 
					     ia, 
					     ja,
					     infoL, 
					     policy, 
					     ichol_buffer);

  status_cusparse = cusparseDcsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_TRANSPOSE, 
					     n, 
					     nnzA, 
					     descrLichol,
					     a, 
					     ia, 
					     ja,
					     infoLt, 
					     policy, 
					     ichol_buffer);

#else
  // H100 + CUDA 12
  cusparseSpSV_updateMatrix(handle_cusparse,
			    descrLichol,
			    a,
			    CUSPARSE_SPSV_UPDATE_GENERAL);
  status_cusparse = cusparseSpSV_analysis(handle_cusparse, 
					  CUSPARSE_OPERATION_NON_TRANSPOSE,
					  &alpha,
					  matL, 
					  NULL, 
					  NULL, 
					  CUDA_R_64F,
					  CUSPARSE_SPSV_ALG_DEFAULT, 
					  descrLichol,
					  ichol_Lbuffer);
  if (status_cusparse != 0 ) printf("We have a problem SV (0), status: %d \n", status_cusparse ); 
  status_cusparse = cusparseSpSV_analysis(handle_cusparse, 
					  CUSPARSE_OPERATION_TRANSPOSE,
					  &alpha,
					  matL, 
					  NULL, 
					  NULL, 
					  CUDA_R_64F,
					  CUSPARSE_SPSV_ALG_DEFAULT, 
					  descrLtichol,
					  ichol_Ltbuffer);
  if (status_cusparse != 0 ) printf("We have a problem SV (1), status: %d \n", status_cusparse ); 
#endif
#else

  status_cusparse = cusparseScsric02_analysis(handle_cusparse,
					      n, 
					      nnzA, 
					      descrM,
					      a, 
					      ia, 
					      ja, 
					      infoM,
					      policy, 
					      ichol_buffer);
  status_cusparse = cusparseXcsric02_zeroPivot(handle_cusparse, infoM, &structural_zero);

  if (CUSPARSE_STATUS_ZERO_PIVOT == status_cusparse) {
    printf("We have a problem: A(%d,%d) is missing\n", structural_zero, structural_zero);
  }

  /* analyze the solves as well */

  status_cusparse = cusparseScsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE, 
					     n, 
					     nnzA, 
					     descrLichol,
					     a, 
					     ia, 
					     ja,
					     infoL, 
					     policy, 
					     ichol_buffer);

  status_cusparse = cusparseScsrsv2_analysis(handle_cusparse, 
					     CUSPARSE_OPERATION_TRANSPOSE, 
					     n, 
					     nnzA, 
					     descrLichol,
					     a, 
					     ia, 
					     ja,
					     infoLt, 
					     policy, 
					     ichol_buffer);

  /* decompose */
  status_cusparse = cusparseScsric02(handle_cusparse, 
				     n, 
				     nnzA, 
				     descrM,
				     a, 
				     ia, 
				     ja, 
				     infoM, 
				     policy, 
				     ichol_buffer);

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
#if (H100 != 1) 
#if USE_FP64 // it is double
  cusparseDcsrsv2_solve(handle_cusparse, 
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
			ichol_buffer);

  /* solve L'*y = aux_vec1 */
  cusparseDcsrsv2_solve(handle_cusparse, 
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
			ichol_buffer);
#else //sp

  cusparseScsrsv2_solve(handle_cusparse, 
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
			ichol_buffer);

  /* solve L'*y = aux_vec1 */
  cusparseScsrsv2_solve(handle_cusparse, 
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
			ichol_buffer);
#endif
#else //it is H100

  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  cusparseDnVecDescr_t vecT;
  cusparseCreateDnVec(&vecX, prec_data->n,  x, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, prec_data->n,  y, CUDA_R_64F);
  cusparseCreateDnVec(&vecT, prec_data->n,  prec_data->aux_vec1, CUDA_R_64F);
  cusparseSpSV_solve(handle_cusparse, 
		     CUSPARSE_OPERATION_NON_TRANSPOSE,
		     &one, 
		     matL, 
		     vecX, 
		     vecT, 
		     CUDA_R_64F,
		     CUSPARSE_SPSV_ALG_DEFAULT, 
		     descrLichol); 
  cusparseSpSV_solve(handle_cusparse, 
		     CUSPARSE_OPERATION_TRANSPOSE,
		     &one, 
		     matL, 
		     vecT, 
		     vecY, 
		     CUDA_R_64F,
		     CUSPARSE_SPSV_ALG_DEFAULT, 
		     descrLtichol); 
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroyDnVec(vecT);

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
				 const real_type *x, 
				 real_type *result){
  /* compute result = L^{-1}x */
  /* we DO NOT assume anything about L diagonal */
  /* d_x3 = L^(-1)dx2 */

  real_type one = 1.0;

#if (H100 != 1)
#if USE_FP64
  cusparseStatus_t status = cusparseDcsrsv2_solve(handle_cusparse, 
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
						  L_buffer);
  //printf("status after tri solve is %d \n", status);
#else
  cusparseStatus_t status = cusparseScsrsv2_solve(handle_cusparse, 
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
						  L_buffer);
#endif
#else // cuda 12 or H100
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  cusparseCreateDnVec(&vecX, n, (double *) x, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, n, result, CUDA_R_64F);
  cusparseSpSV_solve(handle_cusparse, 
		     CUSPARSE_OPERATION_NON_TRANSPOSE,
		     &one, 
		     matL, 
		     vecX, 
		     vecY, 
		     CUDA_R_64F,
		     CUSPARSE_SPSV_ALG_DEFAULT, 
		     descrL); 
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);

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
#if (H100 != 1)
#if USE_FP64 
  cusparseDcsrsv2_solve(handle_cusparse, 
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
			U_buffer);
#else
  cusparseScsrsv2_solve(handle_cusparse, 
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
			U_buffer);
#endif
#else // cuda 12 or H100
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  cusparseCreateDnVec(&vecX, n, (double *) x, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, n, result, CUDA_R_64F);
  cusparseSpSV_solve(handle_cusparse, 
		     CUSPARSE_OPERATION_NON_TRANSPOSE,
		     &one, 
		     matU, 
		     vecX, 
		     vecY, 
		     CUDA_R_64F,
		     CUSPARSE_SPSV_ALG_DEFAULT, 
		     descrU); 
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);


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

  cudaMemcpy(dest, src, sizeof(real_type) * n, cudaMemcpyDeviceToDevice);
}

void cuda_vec_zero(const int n, real_type *vec){

  cuda_vec_zero_kernel<<<1024, 1024>>>(n, vec);
}

