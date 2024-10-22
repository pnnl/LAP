//Written by KS, Mar 2022
//vanilla C version of Laplacian solver.


#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include "common.h"
//Gauss-Seidel, classic version

__global__ void squareofDTimesX(const int n,
    const real_type * dd,
    const real_type *x,
    real_type *y){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  real_type temp;
  while (idx < n){
    temp = dd[idx];
    if (temp <0) temp *=(-1.0f);
    y[idx] =  x[idx]*sqrt(temp);
  }
  idx += blockDim.x * gridDim.x;
}


// needed for easy sorting
struct indexPlusValue
{
  real_type value;
  int idx;
};

//neded for qsort

static int indexPlusValue_comp(const void *a, const void *b)
{
  const struct indexPlusValue *da = (indexPlusValue *)a;
  const struct indexPlusValue *db = (indexPlusValue *)b;

  return da->idx < db->idx ? -1 : da->idx > db->idx;
}

typedef struct
{
  int *coo_rows;
  int *coo_cols;
  real_type *coo_vals;

  int *csr_ia;
  int *csr_ja;
  real_type *csr_vals;

  int n;
  int m;
  int nnz;
  int nnz_unpacked; //nnz in full matrix;
} mmatrix;
//read the matrix (into messy COO)

void read_mm_file(const char *matrixFileName, mmatrix *A)
{
  // this reads triangular matrix but expands into full as it goes (important)
  int noVals = 0;
  FILE *fpm = fopen(matrixFileName, "r");

  char lineBuffer[256];
  //first line, should start with "%%"
  fgets(lineBuffer, sizeof(lineBuffer), fpm);
  char * s = strstr(lineBuffer, "pattern");
  if (s != NULL) noVals =1; 
  while (lineBuffer[0] == '%'){ 
    //printf("Still wrong line: %s \n", lineBuffer);
    fgets(lineBuffer, sizeof(lineBuffer), fpm);
  }

  //first line is size and nnz, need this info to allocate memory
  sscanf(lineBuffer, "%ld %ld %ld", &(A->n), &(A->m), &(A->nnz));
  printf("Matrix size: %d x %d, nnz %d \n",A->n, A->m, A->nnz );
  //allocate

  A->coo_vals = (real_type *)calloc(A->nnz+A->n, sizeof(real_type));
  A->coo_rows = (int *)calloc(A->nnz+A->n, sizeof(int));
  A->coo_cols = (int *)calloc(A->nnz+A->n, sizeof(int));
#if 1
  //read
  int r, c;
  real_type val;
  int i = 0;
  while (fgets(lineBuffer, sizeof(lineBuffer), fpm) != NULL)
  {
    if (noVals == 0){
      sscanf(lineBuffer, "%d %d %lf", &r, &c, &val);
      A->coo_vals[i] = val;
    }else {
      sscanf(lineBuffer, "%d %d", &r, &c);
      A->coo_vals[i] = 1.0;
    }    

    A->coo_rows[i] = r - 1;
    A->coo_cols[i] = c - 1;
    i++;
    if ((c < 1) || (r < 1))
      printf("We have got A PROBLEM! %d %d %16.16f \n", r - 1, c - 1, val);
  }//while
  //main diagonal of A is 0; but L = D-A  so it is not 0 in the Laplacian.
  //this is done to avoid updating CSR pattern
  for (int j=0; j<A->n; ++j){
    A->coo_rows[i] = j;
    A->coo_cols[i] = j;
    A->coo_vals[i] = 1.0f;
    i++;
  } 
  A->nnz+=A->n;
  fclose(fpm);
#endif
}


//COO to CSR
//usual stuff

void coo_to_csr(mmatrix *A)
{
  //this is diffucult
  //first, decide how many nnz we have in each row
  int *nnz_counts;
  nnz_counts = (int *)calloc(A->n, sizeof(int));
  int nnz_unpacked = 0;
  for (int i = 0; i < A->nnz; ++i)
  {
    nnz_counts[A->coo_rows[i]]++;
    nnz_unpacked++;
    if (A->coo_rows[i] != A->coo_cols[i])
    {
      nnz_counts[A->coo_cols[i]]++;
      nnz_unpacked++;
    }
  }
  //allocate full CSR structure
  A->nnz_unpacked = nnz_unpacked;
  A->csr_vals = (real_type *)calloc(A->nnz_unpacked, sizeof(real_type));
  A->csr_ja = (int *)calloc(A->nnz_unpacked, sizeof(int));
  A->csr_ia = (int *)calloc((A->n) + 1, sizeof(int));
  indexPlusValue *tmp = (indexPlusValue *)calloc(A->nnz_unpacked, sizeof(indexPlusValue));
  //create IA (row starts)
  A->csr_ia[0] = 0;
  for (int i = 1; i < A->n + 1; ++i)
  {
    A->csr_ia[i] = A->csr_ia[i - 1] + nnz_counts[i - 1];
  }

  int *nnz_shifts = (int *)calloc(A->n, sizeof(int));
  int r, start;

  for (int i = 0; i < A->nnz; ++i)
  {
    //which row
    r = A->coo_rows[i];
    start = A->csr_ia[r];
    if ((start + nnz_shifts[r]) > A->nnz_unpacked)
      printf("index out of bounds\n");
    tmp[start + nnz_shifts[r]].idx = A->coo_cols[i];
    tmp[start + nnz_shifts[r]].value = A->coo_vals[i];

    nnz_shifts[r]++;

    if (A->coo_rows[i] != A->coo_cols[i])
    {

      r = A->coo_cols[i];
      start = A->csr_ia[r];

      if ((start + nnz_shifts[r]) > A->nnz_unpacked)
        printf("index out of boubns 2\n");
      tmp[start + nnz_shifts[r]].idx = A->coo_rows[i];
      tmp[start + nnz_shifts[r]].value = A->coo_vals[i];
      nnz_shifts[r]++;
    }
  }
  //now sort whatever is inside rows

  for (int i = 0; i < A->n; ++i)
  {

    //now sorting (and adding 1)
    int colStart = A->csr_ia[i];
    int colEnd = A->csr_ia[i + 1];
    int length = colEnd - colStart;

    qsort(&tmp[colStart], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }

  //and copy
  for (int i = 0; i < A->nnz_unpacked; ++i)
  {
    A->csr_ja[i] = tmp[i].idx;
    A->csr_vals[i] = tmp[i].value;
  }
#if 0	
  for (int i=0; i<A->n; i++){
    printf("this is row %d \n", i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; ++j){ 
      printf("  %d,  ", A->csr_ja[j] );			

    }
    printf("\n");

  }
#endif
  free(nnz_counts);
  free(tmp);
  free(nnz_shifts);
  free(A->coo_cols);
  free(A->coo_rows);
  free(A->coo_vals);

}

void create_L_and_split(mmatrix *A, mmatrix *L, mmatrix *U,mmatrix *D, int weighted){
  // we need access to L, U, and D explicitely, not only to the Laplacian
  // w decides whether weighted (w=1) or not (w=0)
  // we need degree of every row
  // allocate L and U bits and pieces;
  L->csr_ia = (int *) calloc (A->n+1, sizeof(int));
  U->csr_ia = (int *) calloc (A->n+1, sizeof(int));
  D->csr_ia = (int *) calloc (A->n+1, sizeof(int));


  L->csr_ja = (int *) calloc (A->nnz-A->n, sizeof(int));
  U->csr_ja = (int *) calloc (A->nnz-A->n, sizeof(int));
  D->csr_ja = (int *) calloc (A->n, sizeof(int));


  L->csr_vals = (real_type *) calloc (A->nnz-A->n, sizeof(real_type));
  U->csr_vals = (real_type *) calloc (A->nnz-A->n, sizeof(real_type));
  D->csr_vals = (real_type *) calloc (A->n, sizeof(real_type));

  int *DD = (int*) calloc(A->n, sizeof(int));
  int iu =0, il=0;
  int col;
  for (int i=0; i<A->n; ++i){
// this only works for unweighted graphs!

//    DD[i] = A->csr_ia[i+1]-A->csr_ia[i];
    //dont count yourself
  //  DD[i]--; 
DD[i] = 0.0;
for (int j = csr_ia[i]; j < csr_ja[i]; ++j){
DD[i] += csr_ja[j];
}  
 }
  //    printf("vertex %d has degree %d \n", i, DD[i]);
  real_type Dsqrt;
  for (int i=0; i<A->n; ++i){
    L->csr_ia[i] = il;
    U->csr_ia[i] = iu;
    if (weighted) Dsqrt = 1.0f/sqrt(DD[i]);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; ++j){
      col = A->csr_ja[j];
      if (col == i) {
        if (!weighted){
          A->csr_vals[j]=(real_type) DD[i]; 
          D->csr_vals[i] = A->csr_vals[j];
          D->csr_ia[i] = i;
          D->csr_ja[i] = i;
        }
        else {
          //printf("Weighted, putting 1.0 on the diagonal \n");
          A->csr_vals[j]=(real_type)1.0f; 
          D->csr_vals[i] = A->csr_vals[j];
          D->csr_ia[i] = i;
          D->csr_ja[i] = i;
        }     
      }
      else{
        if (!weighted){
          A->csr_vals[j] = (-1)*A->csr_vals[j];
        }
        else {

          A->csr_vals[j] = (-1.0f)*A->csr_vals[j]*Dsqrt*(1.0f/sqrt(DD[col]));
          //printf("Weighted, putting %f in (%d, %d) \n", A->csr_vals[j], i, j);
        }
      }

      if (i<col){//row< col, upper part
        U->csr_ja[iu] = A->csr_ja[j];
        U->csr_vals[iu] = A->csr_vals[j];
        iu++;
      }
      if (i>col){//row > col, lower part

        L->csr_ja[il] = A->csr_ja[j];
        L->csr_vals[il] = A->csr_vals[j];
        il++;
      }
    }//for with i

  }//for with j
  D->csr_ia[A->n] = A->n;
  L->csr_ia[A->n] = il;
  U->csr_ia[A->n] = iu;
  L->n = A->n;
  L->m = A->m;
  L->nnz = A->nnz-A->n;

  U->n = A->n;
  U->m = A->m;
  U->nnz = A->nnz-A->n;

  D->n = A->n;
  D->m = A->m;
  D->nnz = A->n;

#if 0	
  printf("\n\n ==== A ==== \n");
  for (int i=0; i<A->n; i++){
    printf("this is row %d \n", i);
    for (int j=A->csr_ia[i]; j<A->csr_ia[i+1]; ++j){ 
      printf("  (%d,%f)  ", A->csr_ja[j], A->csr_vals[j] );			

    }
    printf("\n");

  }

  printf("\n\n ==== D ==== \n");
  for (int i=0; i<D->n; i++){
    printf("this is row %d \n", i);
    for (int j=D->csr_ia[i]; j<D->csr_ia[i+1]; ++j){ 
      printf("  (%d,%f)  ", D->csr_ja[j], D->csr_vals[j] );			

    }
    printf("\n");

  }

  printf("\n\n ==== L ==== \n");
  for (int i=0; i<L->n; i++){
    printf("this is row %d with %d = %d - %d entries \n", i, L->csr_ia[i+1]-L->csr_ia[i], L->csr_ia[i+1], L->csr_ia[i]);
    for (int j=L->csr_ia[i]; j<L->csr_ia[i+1]; ++j){ 
      printf("  %d,  ", L->csr_ja[j] );			

    }
    printf("\n");

  }
  printf("\n\n ==== U ==== \n");
  for (int i=0; i<U->n; i++){
    printf("this is row %d \n", i);
    for (int j=U->csr_ia[i]; j<U->csr_ia[i+1]; ++j){ 
      printf("  %d,  ", U->csr_ja[j] );			

    }
    printf("\n");

  }
#endif
}


int main(int argc, char *argv[])
{
  srand(12345);
  char const *const matrixFileName = argv[1];
  mmatrix *A, *L, *U, *D;

  A = (mmatrix *)calloc(1, sizeof(mmatrix));
  L = (mmatrix *)calloc(1, sizeof(mmatrix));
  U = (mmatrix *)calloc(1, sizeof(mmatrix));
  D = (mmatrix *)calloc(1, sizeof(mmatrix));
  read_mm_file(matrixFileName, A);
  coo_to_csr(A); 

  int weighted = 1;
  create_L_and_split(A,L, U,D, weighted);
  //at this point we know our LAPLACIAN !
  //NOTE: Laplacian is stored in A= L+U+D (matrix splitting).
  //DONT CONFUSE degree matrix with D (D is a diagonal of A) and L with Laplacian (L is lower triangular part of A)

  //allocate space for the GPU
  real_type *d_e, *d_etilde, *d_b, *d_d;

  real_type *e = (real_type *) calloc (A->n, sizeof(real_type));
  real_type *b = (real_type *) calloc (A->n, sizeof(real_type));
  //vector of vertex degrees
  real_type *d = (real_type *) calloc (A->n, sizeof(real_type));
  for (int i=0; i<A->n; ++i) {
    e[i]= 1.0f;
    b[i] =(real_type) (rand()%200)/(rand()%100);
    d[i] = A->csr_ia[i+1]-A->csr_ia[i]-1; //dont count yourself
    printf("b[%d] = %f \n", i, b[i]);
  }

  //create an rhs.

  cudaMalloc(&d_e, A->n * sizeof(real_type));
  cudaMalloc(&d_b, A->n * sizeof(real_type));
  cudaMalloc(&d_d, A->n * sizeof(real_type));

  cudaMemcpy(d_e, e, sizeof(real_type) * A->n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(real_type) * A->n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d, sizeof(real_type) * A->n, cudaMemcpyHostToDevice);
  real_type norme = (real_type) sqrt(A->n);  
  real_type one_over_norme = 1./norme;
  printf ("scaling e by %16.16f, norme %16.16e \n", one_over_norme, norme);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  if (weighted == 0){
    real_type be;
    /* e = (1/norme) e;*/
    cublasDscal(cublas_handle, A->n, &one_over_norme, d_e,1);
    /* be = b'*e*/
    cublasDdot (cublas_handle,A->n,d_e, 1, d_b,1, &be);
    printf("dot product is %16.16f \n", be);
    /*b = b-be*e; */
    be = (-1.0f) * be;
    cublasDaxpy(cublas_handle,A->n, &be,d_e, 1, d_b, 1);
  }else {
    //weighted version
    real_type *d_De;
    real_type *d_D_csr_a; 
    int * d_D_csr_ia, *d_D_csr_ja;

    cudaMalloc(&d_De, A->n * sizeof(real_type));


    cudaMalloc(&d_D_csr_a, A->n * sizeof(real_type));


    //d_De = sqrt(D)*e
    squareofDTimesX<<<A->n/1024+1, 1024>>>(A->n,
        d_d,
        d_e,
        d_De);
    //De_norm = norm(D_De);
    real_type De_norm;

    cublasDdot (cublas_handle,A->n,d_De, 1, d_De,1, &De_norm);
    De_norm = sqrt(De_norm);

    //De = (1/norm(De))*De;

    cublasDscal(cublas_handle, A->n, &De_norm, d_De,1);

    //   bwe = b'*De;
    real_type bwe;

    cublasDdot (cublas_handle,A->n,d_De, 1, d_b,1, &bwe);
    //bProjw = b- bwe*wetilde;
    bwe *= (-1.0f);
    cublasDaxpy(cublas_handle,A->n, &bwe,d_De, 1, d_b, 1);
  }
  // at this point the Laplacian and the rhs are created.




  return 0;
}
