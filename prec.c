#include "common.h"
#include <string.h>

#include "blas.h"
/* dispatcher function */
void prec_function(int *ia,
                   int *ja,
                   real_type *a,
                   int nnzA,
                   pdata *prec_data, 
                   real_type *x, 
                   real_type *y){

  if (strcmp(prec_data->prec_op, "it_jacobi") == 0){
    it_jacobi(ia, ja, a, nnzA, prec_data, x, y);
  }

  if (strcmp(prec_data->prec_op, "line_jacobi") == 0){
    line_jacobi(ia, ja, a, nnzA, prec_data, x, y);
  }

  if (strcmp(prec_data->prec_op, "GS_std") == 0){
    /* std GS */
    GS_std(ia, ja, a, nnzA, prec_data, x, y);
  }

  if (strcmp(prec_data->prec_op, "GS_it") == 0){
    /* iterative GS */
    GS_it(ia, ja, a, nnzA, prec_data, x, y);
  }

  if (strcmp(prec_data->prec_op, "ichol") == 0){
    /* incomplete Cholesky */
    ichol(ia, ja, a, nnzA, prec_data, x, y);
  }
  
  if (strcmp(prec_data->prec_op, "GS_it2") == 0){
    /* iterative GS v2 */
    GS_it2(ia, ja, a, nnzA, prec_data, x, y);
  }

  if(strcmp(prec_data->prec_op, "none") == 0){
    vec_copy(prec_data->n, x, y);
  }
}
