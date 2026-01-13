// based on nvgraph

#include "common.h"
#include "blas.h"
#include "devMem.h"

void cgs2(const int N,
          const int k,
          real_type *X,
          real_type *R) {
/* MATLAB
Q(:,1) = V(:,1)/norm(V(:,1));
R(1,1) = norm(V(:,1));
for i=2:c
    a1 = Q(:,1:i-1)'*V(:,i);
    Q(:,i) = V(:,i) - Q(:,1:i-1)*a1;
     a2 = Q(:,1:i-1)'*Q(:,i);
    Q(:,i) = Q(:,i) - Q(:,1:i-1)*a2;
    R(1:i-1,i) = a1+a2;
    nrm = norm(Q(:,i));
    Q(:,i) = Q(:,i)/nrm;
    R(i,i) = nrm;
end

end

*/
  real_type nrm = dot(N, X, X);
  nrm = sqrt(nrm);
  scal(N, 1.0 / nrm, X);
  real_type *a1;
  real_type one = 1.0;
  real_type zero = 0.0;
  R[0] = nrm;
  // for temp results
  a1 = (real_type *) mallocForDevice(a1, k, sizeof(real_type));
  for (int i = 1; i < k; ++i) {
    gemv("T",
         N,
         i - 1,
         &one,
         X,
         N,
         &X[i * N],
         &zero,
         &R[i * k]);
  }
}

void lobpcg(int n,
            real_type nnz,
            int *ia,
            int *ja,
            real_type *a,
            real_type *X,
            real_type tol,
            pdata *prec_data,
            int k,
            int maxit,
            int *it,
            real_type *eig_vecs,
            real_type *eig_vals) {

#if (CUDA || HIP)
  //allocate data needed for the GPU
  real_type *AX;
  real_type *BX;
  real_type *X_local;
  AX = (real_type *) mallocForDevice(AX, n * k, sizeof(real_type));
  BX = (real_type *) mallocForDevice(BX, n * k, sizeof(real_type));
  X_local = (real_type *) mallocForDevice(X_local, n * k, sizeof(real_type));

#endif
  //initialize X

  randomInit(X, n * k);

  // STEP 1: AX = A*X;
  // use SpGemm in cuda
  // STEP 2

  //
}
