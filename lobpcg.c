// based on nvgraph

#include "common.h"
#include "blas.h"
#include "devMem.h"

void lobpcg(int n, 
    real_type nnz,
		int *ia, //matrix csr data
		int *ja,
		real_type *a,
		real_type tol, //DONT MULTIPLY BY NORM OF B
		pdata * prec_data, //preconditioner data: all Ls, Us etc
    int k, // number of eigenvalues wanted
		int maxit,
		int *it, //output: iteration
    real_type * eig_vecs,
    real_type * eig_vals		
){

#if (CUDA || HIP)
//allocate data needed for the GPU
  real_type * AX;
  real_type * BX;
  real_type * X;
	AX = (real_type*) mallocForDevice(AX, n*k, sizeof(real_type));
	BX = (real_type*) mallocForDevice(BX, n*k, sizeof(real_type));
	X = (real_type*) mallocForDevice(X, n*k, sizeof(real_type));

#endif
//initialize X

randomInit(X, n*k);

// STEP 1: AX = A*X;
// use SpGemm in cuda
// STEP 2

//
}
