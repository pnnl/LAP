#include "common.h"
#include "blas.h"
void line_jacobi(int *ia, int *ja, double *a,int nnzA,  pdata* prec_data, double *vec_in, double *vec_out){
  int n = prec_data->n;
  vec_vec(n, prec_data->d_r, vec_in, vec_out);

}
