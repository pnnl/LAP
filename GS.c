//std Gauss Seidel with tri solves
#include "common.h"
#include "blas.h"
#if (CUDA || HIP)
#include "devMem.h"
#endif

void GS_std(int *ia, int *ja, real_type *a, int nnzA,  pdata *prec_data, real_type *vec_in, real_type *vec_out){

  int n = prec_data->n;
  int k = prec_data->k;
  real_type one = 1.0;
  real_type minusone = -1.0;
  vec_zero(n, vec_out);
  /* backward sweep */
  for (int i = 0; i < k; ++i) {
    /* x = x + L \ ( b - As*x );*/
    //  printf("Inside GS, L solve, before mv: %f \n", dot(n, vec_out, vec_out)); 
    vec_copy(n, vec_in, prec_data->aux_vec2);
    csr_matvec(n, nnzA,ia,  ja,  a, vec_out,  prec_data->aux_vec2, &minusone, &one, "A");
    //  printf("after mv: %f \n", dot(n, prec_data->aux_vec1, prec_data->aux_vec1)); 
    /* aux_vec2 = aux_vec1*(-1) +vec_in */

    /* tri solve L^{-1}*aux_vec2 */
    // printf(" norm r sq: %f \n", dot(n, prec_data->aux_vec2, prec_data->aux_vec2)); 
    lower_triangular_solve(n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la,prec_data->d, prec_data->aux_vec2, prec_data->aux_vec1);
    // printf(" norm sq after L ts: %f \n", dot(n, prec_data->aux_vec1, prec_data->aux_vec1)); 

    axpy(n, 1.0, prec_data->aux_vec1, vec_out);
  }

  //forward sweep
  for (int i = 0; i < k; ++i) {
    /* x = x + L \ ( b - As * x ); */
    /* prec_data->aux_vec1 = A*vec_out */

    vec_copy(n, vec_in, prec_data->aux_vec2);
    csr_matvec(n, nnzA,ia,  ja,  a, vec_out,  prec_data->aux_vec2, &minusone, &one, "A");

    /* tri solve U^{-1}*aux_vec2 */
    upper_triangular_solve(n, prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua,prec_data->d, prec_data->aux_vec2, prec_data->aux_vec1);

    axpy(n, 1.0, prec_data->aux_vec1, vec_out);
  }
}

/* Batched helper: element-wise multiply each column of X by vector d */
static void batch_vec_vec(int n, int k, const real_type *X, const real_type *d, real_type *Y) {
  for (int j = 0; j < k; ++j) {
    vec_vec(n, X + j * n, (real_type*)d, Y + j * n);
  }
}

/* Batched helper: copy matrix */
static void batch_vec_copy(int n, int k, const real_type *src, real_type *dest) {
  for (int j = 0; j < k; ++j) {
    vec_copy(n, (real_type*)(src + j * n), dest + j * n);
  }
}

/* Batched helper: axpy for each column */
static void batch_axpy(int n, int k, real_type alpha, const real_type *X, real_type *Y) {
  for (int j = 0; j < k; ++j) {
    axpy(n, alpha, (real_type*)(X + j * n), Y + j * n);
  }
}

/* Batched sparse matrix-matrix multiply: C = alpha * A * B + beta * C */
static void batch_csr_matvec(int n, int nnz, int *ia, int *ja, real_type *a,
                             int k, const real_type *B, real_type *C,
                             real_type alpha, real_type beta, const char *kind) {
  csrmm(n, k, nnz, ia, ja, a, B, C, alpha, beta, kind);
}

/* Batched iterative GS - process k_batch vectors at once */
void GS_it_batched(int *ia, int *ja, real_type *a, int nnzA, pdata *prec_data,
                   int k_batch, real_type *mat_in, real_type *mat_out) {
  int n = prec_data->n;
  int k = prec_data->k;
  int m = prec_data->m;

  real_type one = 1.0;
  real_type minusone = -1.0;

  /* Allocate batched aux matrices (static for reuse) */
  static real_type *aux_mat1 = NULL;
  static real_type *aux_mat2 = NULL;
  static real_type *aux_mat3 = NULL;
  static int aux_size = 0;

  if (n * k_batch > aux_size) {
    if (aux_mat1) { free(aux_mat1); free(aux_mat2); free(aux_mat3); }
#if (CUDA || HIP)
    aux_mat1 = (real_type*) mallocForDevice(aux_mat1, n * k_batch, sizeof(real_type));
    aux_mat2 = (real_type*) mallocForDevice(aux_mat2, n * k_batch, sizeof(real_type));
    aux_mat3 = (real_type*) mallocForDevice(aux_mat3, n * k_batch, sizeof(real_type));
#else
    aux_mat1 = (real_type*) malloc(n * k_batch * sizeof(real_type));
    aux_mat2 = (real_type*) malloc(n * k_batch * sizeof(real_type));
    aux_mat3 = (real_type*) malloc(n * k_batch * sizeof(real_type));
#endif
    aux_size = n * k_batch;
  }

  /* Zero output */
  vec_zero(n * k_batch, mat_out);

  /* Outer loop */
  for (int j = 0; j < m; ++j) {
    /* r = b - A*x for all columns */
    batch_vec_copy(n, k_batch, mat_in, aux_mat2);
    batch_csr_matvec(n, nnzA, ia, ja, a, k_batch, mat_out, aux_mat2, minusone, one, "A");

    /* y = D^{-1} * r for all columns */
    batch_vec_vec(n, k_batch, aux_mat2, prec_data->d_r, aux_mat1);

    /* Forward sweep: k iterations */
    for (int i = 0; i < k; ++i) {
      batch_vec_copy(n, k_batch, aux_mat2, aux_mat3);
      batch_csr_matvec(n, prec_data->lnnz, prec_data->lia, prec_data->lja, prec_data->la,
                       k_batch, aux_mat1, aux_mat3, minusone, one, "L");
      batch_vec_vec(n, k_batch, aux_mat3, prec_data->d_r, aux_mat1);
    }

    /* Backward sweep: k iterations */
    for (int i = 0; i < k; ++i) {
      batch_vec_copy(n, k_batch, aux_mat2, aux_mat3);
      batch_csr_matvec(n, prec_data->unnz, prec_data->uia, prec_data->uja, prec_data->ua,
                       k_batch, aux_mat1, aux_mat3, minusone, one, "U");
      batch_vec_vec(n, k_batch, aux_mat3, prec_data->d_r, aux_mat1);
    }

    /* Update: x = x + y for all columns */
    batch_axpy(n, k_batch, one, aux_mat1, mat_out);
  }
}

//iterative GS v1
void GS_it(int *ia, int *ja, real_type *a,int nnzA, pdata *prec_data, real_type *vec_in, real_type *vec_out){

  int n = prec_data->n;
  int k = prec_data->k;
  int m = prec_data->m;

  real_type one = 1.0;
  real_type minusone = -1.0;

  /* set vec_out to 0 */
  vec_zero(n, vec_out); 
  //outer loop
  for (int j = 0; j < m; ++j) {
    //r = b - A*x
    vec_copy(n, vec_in, prec_data->aux_vec2);

    csr_matvec(n, nnzA,ia,  ja,  a, vec_out,  prec_data->aux_vec2, &minusone, &one, "A");
    // printf("res nrm %16.16f \n", dot(n,  prec_data->aux_vec2,  prec_data->aux_vec2));  
    /* r = aux_vec2 = aux_vec1*(-1) +vec_in */
    // y = aux_vec1 = D^{-1}aux_vec2
    vec_vec(n, prec_data->aux_vec2, prec_data->d_r, prec_data->aux_vec1); 
    // printf("res nrm after scaling %16.16f \n", dot(n,  prec_data->aux_vec1,  prec_data->aux_vec1));  
    for (int i = 0; i < k; ++i) {
      /* y = v.*(r-L*y); */
      /* vec3 = L*vec1 */   
      vec_copy(n, prec_data->aux_vec2, prec_data->aux_vec3);      
      csr_matvec(n, prec_data->lnnz,prec_data->lia,prec_data->lja,  prec_data->la, prec_data->aux_vec1, prec_data->aux_vec3, &minusone, &one, "L");
      /* axpy */

      // printf("\t res nrm inside first loop, after mv  %f \n", dot(n,   prec_data->aux_vec3,  prec_data->aux_vec3));  
      vec_vec(n, prec_data->aux_vec3, prec_data->d_r, prec_data->aux_vec1); 
    }

    // printf("res nrm after first loop %f \n", dot(n,  prec_data->aux_vec1,  prec_data->aux_vec1));  
    for (int i = 0; i < k; ++i) {
      /* y = v.*(r-L*y); */
      vec_copy(n, prec_data->aux_vec2, prec_data->aux_vec3);      
      csr_matvec(n, prec_data->unnz,prec_data->uia,prec_data->uja,  prec_data->ua, prec_data->aux_vec1, prec_data->aux_vec3, &minusone, &one, "U");
      /* axpy */
      vec_vec(n, prec_data->aux_vec3, prec_data->d_r, prec_data->aux_vec1); 
    }

    // printf("res nrm after second %f \n", dot(n,  prec_data->aux_vec1,  prec_data->aux_vec1));  
    /* vec_out = vec_out + vec1 */ 
    axpy(n, 1.0, prec_data->aux_vec1, vec_out);
    // printf("res nrm after update  %f \n", dot(n,  vec_out, vec_out));  
  }
}

//iterative GS v2
void GS_it2(int *ia, int *ja, real_type *a,int nnzA, pdata *prec_data, real_type *vec_in, real_type *vec_out){

  int n = prec_data->n;
  int k = prec_data->k;
  int m = prec_data->m;

  real_type one = 1.0;
  real_type zero = 0.0;  
  real_type minusone = -1.0; 
  /* y = Dinv.*b; */
  vec_vec(n, vec_in, prec_data->d_r, prec_data->aux_vec1); 
  /* outer loop */
  for (int j = 0; j < m; ++j) {

    /* inner loop 1 */
for (int i = 0; i < 1; ++i){
      /* L*(Dinv*b) */
      csr_matvec(n, prec_data->lnnz,prec_data->lia,  prec_data->lja,  prec_data->la, prec_data->aux_vec1,  prec_data->aux_vec2, &one, &zero, "L");
      /* U*(Dinv*b) */
      csr_matvec(n, prec_data->unnz,prec_data->uia,  prec_data->uja,  prec_data->ua, prec_data->aux_vec1,  prec_data->aux_vec3, &one, &zero, "U");
      /* (U+L)Dinv*b */
      axpy(n, 1.0, prec_data->aux_vec3, prec_data->aux_vec2);
      vec_copy(n, vec_in, prec_data->aux_vec3);
      /* aux3  = vec_in-1.0*(U+L)Dinv*b */
      axpy(n, -1.0, prec_data->aux_vec2, prec_data->aux_vec3);
      /* scale */
      vec_vec(n, prec_data->aux_vec3, prec_data->d_r, prec_data->aux_vec2);
    } /* inner loop 1 */
  /* compute residual:  r = b - L*y; */
    /* vec3 = b */
    vec_copy(n, vec_in, prec_data->aux_vec3);
    /* vec1 = L*y = L*vec2 */
    /* r = b-L*y : vec2 =  vec2 - vec1 */
    csr_matvec(n, prec_data->lnnz, prec_data->lia,  prec_data->lja,  prec_data->la, prec_data->aux_vec2,  prec_data->aux_vec3, &minusone, &one, "L");

    /* inner loop 2 */
    for (int i = 0; i < k; ++i){
      /* y = (v).* ( r - U * y ); */
      /* vec1 = U*vec2 = U*y */
      vec_copy(n, prec_data->aux_vec3, prec_data->aux_vec1);
      csr_matvec(n, prec_data->unnz,prec_data->uia,  prec_data->uja,  prec_data->ua, prec_data->aux_vec2,  prec_data->aux_vec1, &minusone, &one, "U");
      /* scale */
      vec_vec(n, prec_data->d_r, prec_data->aux_vec1, prec_data->aux_vec2);
    } /* inner loop 2 */
    /* residual again */
    /* r = b - U*y; */
    /* vec3 = b */ 
    vec_copy(n, vec_in, prec_data->aux_vec3);
    /* vec1 = U*y = U*vec2 */
    csr_matvec(n, prec_data->unnz,prec_data->uia,  prec_data->uja,  prec_data->ua, prec_data->aux_vec2,  prec_data->aux_vec3, &minusone, &one, "U");

    /* inner loop 3 */
    for (int i = 0; i < k; ++i){
      /* y = (v).* ( r - L * y ); */
      /* vec1 = L*vec2 = L*y */
      vec_copy(n, prec_data->aux_vec3, prec_data->aux_vec1);
      csr_matvec(n, prec_data->lnnz,prec_data->lia,  prec_data->lja,  prec_data->la, prec_data->aux_vec2,  prec_data->aux_vec1, &minusone, &one, "L");
      vec_vec(n, prec_data->d_r, prec_data->aux_vec1, prec_data->aux_vec2);
    } /* inner loop 3 */
    vec_copy(n, prec_data->aux_vec2,prec_data->aux_vec1);

  }/* outer loop */

  vec_copy(n, prec_data->aux_vec2, vec_out);
}

