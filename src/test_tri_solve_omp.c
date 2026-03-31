
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() 
{
  int n = 5;
  int nnz = 5;
  int num_devices = omp_get_num_devices();


  printf("Number of available devices %d\n", num_devices);
  double  ua[5] = {2.0,3.0,1.0,2.0,1.0};
  int uja[5] = {1,2,3,4,4};
  int uia[6] = {0,2, 2,4, 5,5};
  double x[5] = {8.0,8.0, 8.0, 8.0, 8.0}; 
  double diagonal[5] = {1.0, 7.0, 4.0, 3.0, 9.0};
  double * result = (double *) calloc(n, sizeof(double));
  // int * uja = (int *) calloc(N, sizeof(double));
  int i, j, col; 

  double start_time = omp_get_wtime();
#pragma omp target teams distribute map(to:ua[0:nnz], uja[0:nnz],ua[0:n+1],x[0:n])  map(tofrom:result[0:n])
  for (i=n-1; i>=0; --i){
    result[i] = 0.0f;
    double s=0.0;

#pragma omp simd private(j) reduction(+:s)

    for (j=uia[i]; j<uia[i+1]; ++j){
      col = uja[j];
      s += (-1.0)*ua[j]*result[col];
    } 
    {
      result[i] =(s+x[i])/diagonal[i];
    } 
  }

  double time = omp_get_wtime() - start_time;  
  printf("Elapsed (GPU): %g seconds\n", time);
  for (i =0; i<5; ++i)
  {
    printf( " %f\n ", result[i]);
  }


  start_time = omp_get_wtime();
  for (i=n-1; i>=0; --i){
    result[i] = 0.0f;
    double s=0.0;

#pragma omp simd private(j) reduction(+:s)

    for (j=uia[i]; j<uia[i+1]; ++j){
      col = uja[j];
      s += (-1.0)*ua[j]*result[col];
    } 
    {
      result[i] =(s+x[i])/diagonal[i];
    } 
  }

  time = omp_get_wtime() - start_time;  
  printf("Elapsed (NO GPU): %g seconds\n", time);
  for (i =0; i<5; ++i)
  {
    printf( " %f\n ", result[i]);
  }
}



