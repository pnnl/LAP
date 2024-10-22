#include "io_utils.h"
#include "common.h"
/* utilities needed for matrix I/O */
/* and other operations of the kind */
#if CUDA
#include "devMem.h"
#endif

#if HIP
#include "devMem.h"
#endif

/* read the matrix (into UNSORTER COO) */

void read_mm_file(const char *matrixFileName, mmatrix *A) 
{
  /* this reads triangular matrix or general matrix  */
  int noVals = 0;
  int sym = 0;
  FILE *fpm = fopen(matrixFileName, "r");

  char lineBuffer[256];
  /* first line, should start with "%%" */
  fgets(lineBuffer, sizeof(lineBuffer), fpm);

  char *s = strstr(lineBuffer, "pattern");
  if (s != NULL) {
    noVals = 1;
  } 

  s = strstr(lineBuffer, "symmetric");
  if (s != NULL) {
    sym = 1;
  } 
  A->symmetric = sym; 

  // printf("noVals? %d \n", noVals);  
  while (lineBuffer[0] == '%') { 
    fgets(lineBuffer, sizeof(lineBuffer), fpm);
  }

  /* first line is size and nnz, need this info to allocate memory */
  sscanf(lineBuffer, "%ld %ld %ld", &(A->n), &(A->m), &(A->nnz));
  // printf("Matrix size: %d x %d, nnz %d \n",A->n, A->m, A->nnz );

  /* allocate */

  A->coo_vals = (double *) calloc(A->nnz, sizeof(double));
  A->coo_rows = (int *)    calloc(A->nnz, sizeof(int));
  A->coo_cols = (int *)    calloc(A->nnz, sizeof(int));

  /* read */

  int r, c;
  double val;
  int i = 0;
  while (fgets(lineBuffer, sizeof(lineBuffer), fpm) != NULL)
  {
    if (noVals == 0) {
      sscanf(lineBuffer, "%d %d %lf", &r, &c, &val);
      A->coo_vals[i] = val;
    } else {
      sscanf(lineBuffer, "%d %d", &r, &c);
      A->coo_vals[i] = 1.0;
    }    

    A->coo_rows[i] = r - 1;
    A->coo_cols[i] = c - 1;
    i++;
    if ((c < 1) || (r < 1))
      printf("We have got A PROBLEM! %d %d %16.16f \n", r - 1, c - 1, val);
  } /* while */
  fclose(fpm);
}

/* coo to sorted coo */



/* read adjacency matrix and make it into (unweighted) graph Laplacian*/

void read_adjacency_file(const char *matrixFileName, mmatrix *A) 
{

  int noVals = 0;
  FILE *fpm = fopen(matrixFileName, "r");

  char lineBuffer[256];
  /* first line, should start with "%%" */

  fgets(lineBuffer, sizeof(lineBuffer), fpm);
  char *s = strstr(lineBuffer, "pattern");
  if (s != NULL) { 
    noVals = 1;
  }
  A->symmetric = 1; /* adjacency matrix is symmetric unless hypergraph */ 
  while (lineBuffer[0] == '%') { 
    fgets(lineBuffer, sizeof(lineBuffer), fpm);
  }

  /* first line is size and nnz, need this info to allocate memory */
  sscanf(lineBuffer, "%ld %ld %ld", &(A->n), &(A->m), &(A->nnz));

  /* allocate - note extra space for diagonal!*/
  A->coo_vals = (double *) calloc(A->nnz + A->n, sizeof(double));
  A->coo_rows = (int *)    calloc(A->nnz + A->n, sizeof(int));
  A->coo_cols = (int *)    calloc(A->nnz + A->n, sizeof(int));

  int r, c;
  double val;
  int i = 0;
  while (fgets(lineBuffer, sizeof(lineBuffer), fpm) != NULL)
  {
    if (noVals == 0){
      sscanf(lineBuffer, "%d %d %lf", &r, &c, &val);
      A->coo_vals[i] = (real_type) val;
    } else {
      sscanf(lineBuffer, "%d %d", &r, &c);
      A->coo_vals[i] = (real_type) 1.0;
    }    

    A->coo_rows[i] = r - 1;
    A->coo_cols[i] = c - 1;
    i++;
    if ((c < 1) || (r < 1))
      printf("We have got A PROBLEM! %d %d %16.16f \n", r - 1, c - 1, val);
  } /* while */
  /* main diagonal of A is 0; but L = D - A  so it is not 0 in the Laplacian.*/
  /* this is done to avoid updating CSR pattern */
  for (int j = 0; j < A->n; ++j) {
    A->coo_rows[i] = j;
    A->coo_cols[i] = j;
    A->coo_vals[i] = 1.0;
    i++;
  } 
  A->nnz += A->n;
  fclose(fpm);
}

/* COO to CSR */
void coo_to_csr(mmatrix *A) 
{
  /* first, decide how many nnz we have in each row */
  int *nnz_counts;
  nnz_counts = (int *) calloc(A->n, sizeof(int));
  int nnz_unpacked = 0;
  for (int i = 0; i < A->nnz; ++i) {
    nnz_counts[A->coo_rows[i]]++;
    nnz_unpacked++;
    if ((A->coo_rows[i] != A->coo_cols[i]) && (A->symmetric)) {
      nnz_counts[A->coo_cols[i]]++;
      nnz_unpacked++;
    }
  }

  /* allocate full CSR structure */
  A->nnz_unpacked = nnz_unpacked;

  A->csr_vals = (real_type *) calloc(A->nnz_unpacked, sizeof(real_type));
  A->csr_ja = (int *) calloc(A->nnz_unpacked, sizeof(int));
  A->csr_ia = (int *) calloc((A->n) + 1, sizeof(int));
  indexPlusValue *tmp = (indexPlusValue *) calloc(A->nnz_unpacked, sizeof(indexPlusValue));

  /* create IA (row pointers) */
  A->csr_ia[0] = 0;
  for (int i = 1; i < A->n + 1; ++i) {
    A->csr_ia[i] = A->csr_ia[i - 1] + nnz_counts[i - 1];
  }

  int *nnz_shifts = (int *) calloc(A->n, sizeof(int));
  int r, start;

  for (int i = 0; i < A->nnz; ++i) {
    /* which row 8 */ 
    r = A->coo_rows[i];
    start = A->csr_ia[r];
    if ((start + nnz_shifts[r]) > A->nnz_unpacked) {
      printf("index out of bounds\n");
    }

    tmp[start + nnz_shifts[r]].idx = A->coo_cols[i];
    tmp[start + nnz_shifts[r]].value = (real_type) A->coo_vals[i];

    nnz_shifts[r]++;

    if ((A->coo_rows[i] != A->coo_cols[i]) && (A->symmetric)) {

      r = A->coo_cols[i];
      start = A->csr_ia[r];

      if ((start + nnz_shifts[r]) > A->nnz_unpacked) {
        printf("index out of boubns 2\n");
      }

      tmp[start + nnz_shifts[r]].idx = A->coo_rows[i];
      tmp[start + nnz_shifts[r]].value = (real_type) A->coo_vals[i];
      nnz_shifts[r]++;
    }
  }
  /* now sort whatever is inside rows */

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
  for (int i = 0; i < 10; i++) {
    printf("this is row %d \n", i);
    for (int j = A->csr_ia[i]; j < A->csr_ia[i + 1]; ++j) { 
      printf("  (%d, %f)  ", A->csr_ja[j], A->csr_vals[j] );      

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

/* read rhs from file */

void read_rhs(const char *rhsFileName, real_type *rhs) {

  FILE* fpr = fopen(rhsFileName, "r");
  char lineBuffer[256];

  fgets(lineBuffer, sizeof(lineBuffer), fpr);
  while (lineBuffer[0] == '%') { 
    fgets(lineBuffer, sizeof(lineBuffer), fpr);
  } 
  int N, m;
  sscanf(lineBuffer, "%ld %ld", &N, &m);  

  int i = 0;
  double val;
  /* allocate */

  /* if there are multiple rhs in a file, read the first one only, ignore the rest */

  while ((fgets(lineBuffer, sizeof(lineBuffer), fpr) != NULL) && (i != N )) {
    sscanf(lineBuffer, "%lf", &val);
    rhs[i] = (real_type) val;
    //   printf("rhs[%d] = %16.18f \n",i, val);
    i++;
  }
  fclose(fpr);
}

/* Split matrix A = L + U + D */

void split(mmatrix *A, mmatrix *L, mmatrix *U, mmatrix *D) 
{
  /* we need access to L, U, and D explicitely
   * we need degree of every row
   * allocate L and U bits and pieces; */

  L->csr_ia = (int *) calloc (A->n + 1, sizeof(int));
  U->csr_ia = (int *) calloc (A->n + 1, sizeof(int));
  D->csr_ia = (int *) calloc (A->n + 1, sizeof(int));


  L->csr_ja = (int *) calloc (A->nnz - A->n, sizeof(int));
  U->csr_ja = (int *) calloc (A->nnz - A->n, sizeof(int));
  D->csr_ja = (int *) calloc (A->n, sizeof(int));

  L->csr_vals = (real_type *) calloc (A->nnz - A->n, sizeof(real_type));
  U->csr_vals = (real_type *) calloc (A->nnz - A->n, sizeof(real_type));
  D->csr_vals = (real_type *) calloc (A->n, sizeof(real_type));

  int iu = 0, il = 0;
  int col;

  for (int i = 0; i < A->n; ++i) {
    L->csr_ia[i] = il;
    U->csr_ia[i] = iu;
    for (int j = A->csr_ia[i]; j < A->csr_ia[i + 1]; ++j) {
      col = A->csr_ja[j];
      if (col == i) {
        D->csr_vals[i] = A->csr_vals[j];
        D->csr_ia[i] = i;
        D->csr_ja[i] = i;
      }
      if (i < col) { /* row< col, upper part */
        U->csr_ja[iu] = A->csr_ja[j];
        U->csr_vals[iu] = A->csr_vals[j];
        iu++;
      }
      if (i > col) {/* row > col, lower part */
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
  for (int i = 0; i < 10; i++) {
    printf("this is row %d \n", i);
    for (int j = A->csr_ia[i]; j<A->csr_ia[i + 1]; ++j) { 
      printf("  (%d,%f)  ", A->csr_ja[j], A->csr_vals[j]);     
    }
    printf("\n");
  }
  printf("\n\n ==== D ==== \n");
  for (int i=0; i < 10; i++){
    printf("this is row %d \n", i);
    for (int j = D->csr_ia[i]; j < D->csr_ia[i + 1]; ++j){ 
      printf("  (%d,%f)  ", D->csr_ja[j], D->csr_vals[j] );     
    }
    printf("\n");

  }

  printf("\n\n ==== L ==== \n");
  for (int i = 0; i < 10; i++){
    printf("this is row %d with %d = %d - %d entries \n", i, L->csr_ia[i+1]-L->csr_ia[i], L->csr_ia[i+1], L->csr_ia[i]);
    for (int j = L->csr_ia[i]; j < L->csr_ia[i + 1]; ++j){ 
      printf(" (%d,%f)  ", L->csr_ja[j], L->csr_vals[j] );      
    }
    printf("\n");
  }
  printf("\n\n ==== U ==== \n");
  for (int i = 0; i < 10; i++) {
    printf("this is row %d \n", i);
    for (int j = U->csr_ia[i]; j < U->csr_ia[i + 1]; ++j){ 
      printf(" (%d,%f)  ", U->csr_ja[j], U->csr_vals[j] );      
    }
    printf("\n");
  }
#endif
}

/* create LAPLACIAN out of ADJACENCY matrix */ 

void create_L_and_split(mmatrix *A, mmatrix *L, mmatrix *U, mmatrix *D, int weighted) 
{
  /* we need access to L, U, and D explicitely, not only to the Laplacian
   * w decides whether weighted (w == 1) or not (w == 0)
   * we need degree of every row */

  /* allocate L and U bits and pieces; */
  L->csr_ia = (int *) calloc (A->n + 1, sizeof(int));
  U->csr_ia = (int *) calloc (A->n + 1, sizeof(int));
  D->csr_ia = (int *) calloc (A->n + 1, sizeof(int));

  L->csr_ja = (int *) calloc (A->nnz - A->n, sizeof(int));
  U->csr_ja = (int *) calloc (A->nnz - A->n, sizeof(int));
  D->csr_ja = (int *) calloc (A->n, sizeof(int));


  L->csr_vals = (real_type *) calloc (A->nnz - A->n, sizeof(real_type));
  U->csr_vals = (real_type *) calloc (A->nnz - A->n, sizeof(real_type));
  D->csr_vals = (real_type *) calloc (A->n, sizeof(real_type));

  int *DD = (int *) calloc(A->n, sizeof(int));
  int iu = 0, il = 0;
  int col;
  for (int i = 0; i < A->n; ++i) {
    DD[i] = -1.0; /*  dont count yourself */
    for (int j = A->csr_ia[i]; j < A->csr_ia[i+1]; ++j) {
      DD[i] += A->csr_vals[j];
    }  
  }

  real_type Dsqrt;
  for (int i = 0; i < A->n; ++i) {
    L->csr_ia[i] = il;
    U->csr_ia[i] = iu;
    if (weighted) { 
      Dsqrt = 1.0 / sqrt(DD[i]);
    }
    for (int j = A->csr_ia[i]; j < A->csr_ia[i + 1]; ++j) {
      col = A->csr_ja[j];
      if (col == i) {
        if (!weighted){
          A->csr_vals[j] = (real_type) DD[i]; 
          D->csr_vals[i] = A->csr_vals[j];
          D->csr_ia[i] = i;
          D->csr_ja[i] = i;
        } else {
          //printf("Weighted, putting 1.0 on the diagonal \n");
          A->csr_vals[j]= (real_type) 1.0; 
          D->csr_vals[i] = A->csr_vals[j];
          D->csr_ia[i] = i;
          D->csr_ja[i] = i;
        }     
      } else {
        if (!weighted){
          A->csr_vals[j] = (-1.0)*A->csr_vals[j];
        } else {
          A->csr_vals[j] = (-1.0) * A->csr_vals[j] * Dsqrt * (1.0 / sqrt(DD[col]));
        }
      }

      if (i < col) { /* row < col, upper part */
        U->csr_ja[iu] = A->csr_ja[j];
        U->csr_vals[iu] = A->csr_vals[j];
        iu++;
      }
      if (i > col) { /* row > col, lower part */
        L->csr_ja[il] = A->csr_ja[j];
        L->csr_vals[il] = A->csr_vals[j];
        il++;
      }
    } /* for with i */
  } /* for with j */

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
  for (int i = 0; i < 10; i++) {
    printf("this is row %d \n", i);
    for (int j = A->csr_ia[i]; j < A->csr_ia[i + 1]; ++j) { 
      printf("  (%d,%f)  ", A->csr_ja[j], A->csr_vals[j] );			
    }
    printf("\n");
  }

  printf("\n\n ==== D ==== \n");
  for (int i = 0; i < 10; i++) {
    printf("this is row %d \n", i);
    for (int j = D->csr_ia[i]; j < D->csr_ia[i + 1]; ++j) { 
      printf("  (%d,%f)  ", D->csr_ja[j], D->csr_vals[j] );			
    }
    printf("\n");
  }

  printf("\n\n ==== L ==== \n");
  for (int i = 0; i < 10; i++) {
    printf("this is row %d with %d = %d - %d entries \n", i, L->csr_ia[i+1]-L->csr_ia[i], L->csr_ia[i+1], L->csr_ia[i]);
    for (int j = L->csr_ia[i]; j < L->csr_ia[i + 1]; ++j) { 
      printf(" (%d,%f)  ", L->csr_ja[j], L->csr_vals[j] );			
    }
    printf("\n");
  }
  printf("\n\n ==== U ==== \n");
  for (int i = 0; i < 10; i++){
    printf("this is row %d \n", i);
    for (int j = U->csr_ia[i]; j < U->csr_ia[i + 1]; ++j) { 
      printf(" (%d,%f)  ", U->csr_ja[j], U->csr_vals[j] );			
    }
    printf("\n");
  }
#endif
}
