#include "simple_blas.h"
#include <math.h>

real_type simple_dot (const int n, const real_type *v, const real_type *w){
  real_type sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += v[i] * w[i];
  }
  return sum;
}

void simple_scal (const int n, const real_type alpha, real_type *v){
  for (int i = 0; i < n; ++i) {
    v[i] *= alpha;
  }
}

void simple_axpy (const int n, const real_type alpha, const real_type *x, real_type *y){
  for (int i = 0; i < n; ++i){
    y[i] += alpha * x[i];
  }
}

void simple_csr_matvec(const int n,
                       const int nnz, 
                       const int *ia, 
                       const int *ja,
                       const real_type *a, 
                       const real_type *x, 
                       real_type *result,
                       const real_type *al, 
                       const real_type *bet){
  real_type alpha = *al;
  real_type beta = *bet;
  /* go through every row */
  for (int i = 0; i < n; ++i) {
    /* go through each column in this row */
    result[i] *= beta;  
    for (int j = ia[i]; j < ia[i + 1]; j++) {
      int col = ja[j];
      result[i] += (alpha * a[j] * x[col]);
    }
  }
}

void simple_lower_triangular_solve(const int n, 
                                   const int nnz, 
                                   const int *lia, 
                                   const int *lja, 
                                   const real_type *la,
                                   const real_type *diagonal,
                                   const real_type *x, 
                                   real_type *result){
  /* compute result = L^{-1}x */
  /* go through each row (starting from 0) */
  for (int i = 0; i < n; ++i) {
    result[i] = x[i];
    for (int j = lia[i]; j < lia[i + 1]; ++j) {
      int col = lja[j];
      result[i] -= la[j] * result[col];  
    }
    result[i] /= diagonal[i];
  }
}


void simple_upper_triangular_solve(const int n,
                                   const int nnz, 
                                   const int *uia, 
                                   const int *uja, 
                                   const real_type *ua, 
                                   const real_type *diagonal, 
                                   const real_type *x, 
                                   real_type *result){
  /* compute result = U^{-1}x */
  /* go through each row (starting from the last row) */
  for (int i = n-1; i >= 0; --i) {
    result[i] = x[i];
    for (int j = uia[i]; j < uia[i+1]; ++j) {
      int col = uja[j];
      result[i] -= ua[j] * result[col];
    }
    result[i] /= diagonal[i];
  }
}

/* not std blas but needed and embarassingly parallel */ 

/* simple vec-vec computes an element-wise product (needed for scaling) */

void simple_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res){
  for (int i = 0; i < n; ++i) {
    res[i] = x[i] * y[i];
  }
}

/* vector reciprocal computes 1./d */ 

void simple_vector_reciprocal(const int n, const real_type *v, real_type *res){

  for (int i = 0; i < n; ++i){
    if  (v[i] != 0.0) {
      res[i] = 1.0/v[i];
    } else {
      res[i] = 0.0;
    }
  }
}

/* vector sqrt takes an sqrt from each vector entry */

void simple_vector_sqrt(const int n, const real_type *v, real_type *res){

  for (int i = 0; i < n; ++i) {
    if  (v[i] >= 0.0) {
      res[i] = sqrt(v[i]);
    } else {
      res[i] = 0.0;
    }
  }
}

void simple_vec_copy(const int n, const real_type *src, real_type *dest){

  for (int i = 0; i < n; ++i) {
    dest[i] = src[i];  
  }
}


void simple_vec_zero(const int n, real_type *vec){

  for (int i = 0; i < n; ++i) {
    vec[i] = 0.0;  
  }
}


void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a, 
                      int *lia,
                      int *lja,
                      real_type *la)
{
  for (int i = 0; i < n; ++i) {
    /*   
     *   if (n>100000) {
     *      if (i %100==0) printf("processing row: %d\n", i);
     }*/

    a[ia[i]] = sqrt(a[ia[i]]);
    for (int m = ia[i] + 1; m < ia[i + 1]; ++m){
      a[m] = a[m]/a[ia[i]]; 
    }

    for (int m = ia[i] + 1; m < ia[i + 1]; ++m) {
      for (int k = ia[ja[m]]; k < ia[ja[m] + 1]; ++k) {
        for (int l = m; l < ia[i + 1]; ++l) {
          if (ja[l] == ja[k]){ 
            a[k] -= a[m] * a[l];
          } /* if */
        } /* loop with l */
      } /* loop with k */ 
    } /* loop with m */
  }
  /* at this point, what we have in (ia, ja, a) is CSR format of L^T (so the same as "U").
   * and we need L (also in CSR), so we have to transpose. */

  int *Lcounts = (int *) calloc (nnzA, sizeof(int));
  for (int i = 0; i < n; ++i) {
    for (int j = ia[i]; j < ia[i + 1]; ++j) {
      int row = ja[j];
      real_type val = a[j];
      la[lia[row] + Lcounts[row]] = val;
      Lcounts[row]++;
    } 
  }
  free(Lcounts); 
}

void simple_ichol(const int *ia, 
                  const int *ja, 
                  real_type *a, 
                  const int nnzA, 
                  pdata *prec_data, 
                  real_type *x, 
                  real_type *y){
  /* we dont really need A but whatever */
  real_type *la = prec_data->la;
  int *lia = prec_data->lia;
  int *lja = prec_data->lja;
  real_type *ua = prec_data->ua;
  int *uia = prec_data->uia;
  int *uja = prec_data->uja;
  int n = prec_data->n;  

  /* compute result = L^{-1}x */ 
  for (int i = 0; i < n; ++i) {
    prec_data->aux_vec1[i] = x[i];
    for (int j = lia[i]; j < lia[i + 1]; ++j) {
      int col = lja[j];
      if (col != i){
        prec_data->aux_vec1[i] -= la[j] * prec_data->aux_vec1[col]; 
      }
    }
    prec_data->aux_vec1[i] /= la[lia[i + 1] - 1]; ;
  }

  for (int i = n - 1; i >= 0; --i) {
    y[i] = prec_data->aux_vec1[i];
    for (int j = uia[i]; j < uia[i + 1]; ++j) {
      int col = uja[j];
      if (col != i){
        y[i] -= ua[j] * y[col];
      }
    }
    y[i] /= ua[uia[i]]; /*divide by the diagonal entry*/
  }

}
