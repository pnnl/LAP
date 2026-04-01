
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() 
{
  int n = 5;
  int nnz = 10;
  int num_devices = omp_get_num_devices();


  printf("Number of available devices %d\n", num_devices);
  double  a[10] = {1.0, 2.0,3.0,7.0, 4.0, 1.0,2.0,3.0,1.0, 9.0};
  int ja[10] = {0,1,2,1,2,3,4,3,4,4};
  int ia[6] = {0,3, 4,7, 9, 10};
  double x[5] = {1.0,2.0, 3.0, 4.0, 5.0}; 
  double * result = (double *) calloc(n, sizeof(double));
  for (int i=0; i<5; ++i) {result[i] = 1.0;}
  // int * uja = (int *) calloc(N, sizeof(double));
  int i, j, col;
  double s;
  double alpha = -2.0f;
  double beta = 1.5f;
  #pragma omp target teams distribute parallel for  schedule(static) private(i, s)map(to:a[0:nnz], x[0:n], ia[0:n+1], ja[0:nnz], alpha, beta) map(tofrom:result[0:n])
  for (i=0; i<n; ++i){
    //go through each column in this row
    s =    result[i] * beta; 
    #pragma omp simd private(j) reduction(+:s)
    for (j=ia[i]; j<ia[i+1]; j++){
      col = ja[j];
      s += (alpha*a[j]*x[col]);   
 }
    result[i] = s;
  }
  for (i =0; i<5; ++i)
  {
    printf( " %f\n ", result[i]);
  }
}



