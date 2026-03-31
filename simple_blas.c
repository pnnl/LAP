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

void simple_gemv(const char *T,
                 const int m,
                 const int n,
                 const double *alpha,
                 const double *A,
                 const int lda,
                 const double *x,
                 const double *beta,
                 double *y)
{
  /* y = alpha * A * x + beta * y (or A^T if T == "T") */
  double a = *alpha;
  double b = *beta;
  
  if (T[0] == 'T' || T[0] == 't') {
    /* Transpose case: y = alpha * A^T * x + beta * y */
    /* A is m x n, A^T is n x m, x is m-vector, y is n-vector */
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int i = 0; i < m; ++i) {
        sum += A[i + j * lda] * x[i];
      }
      y[j] = a * sum + b * y[j];
    }
  } else {
    /* Non-transpose case: y = alpha * A * x + beta * y */
    /* A is m x n, x is n-vector, y is m-vector */
    for (int i = 0; i < m; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += A[i + j * lda] * x[j];
      }
      y[i] = a * sum + b * y[i];
    }
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

/* GEMM: C = alpha * op(A) * op(B) + beta * C */
/* A is m x k (or k x m if transposed), B is k x n (or n x k if transposed), C is m x n */
/* All matrices in column-major order */
void simple_gemm(const char *transA,
                 const char *transB,
                 const int m,
                 const int n,
                 const int k,
                 const real_type *alpha,
                 const real_type *A,
                 const int lda,
                 const real_type *B,
                 const int ldb,
                 const real_type *beta,
                 real_type *C,
                 const int ldc) {
  real_type a = *alpha;
  real_type b = *beta;
  int ta = (transA[0] == 'T' || transA[0] == 't') ? 1 : 0;
  int tb = (transB[0] == 'T' || transB[0] == 't') ? 1 : 0;
  
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      real_type sum = 0.0;
      for (int l = 0; l < k; ++l) {
        real_type aval = ta ? A[l + i * lda] : A[i + l * lda];
        real_type bval = tb ? B[j + l * ldb] : B[l + j * ldb];
        sum += aval * bval;
      }
      C[i + j * ldc] = a * sum + b * C[i + j * ldc];
    }
  }
}

/* 
 * Simple Cholesky decomposition for positive definite matrix
 * A is n x n symmetric positive definite, stored in column-major
 * On exit, lower triangular part of A contains L such that A = L*L'
 */
static int simple_cholesky(int n, real_type *A) {
  for (int j = 0; j < n; ++j) {
    real_type sum = A[j + j * n];
    for (int k = 0; k < j; ++k) {
      sum -= A[j + k * n] * A[j + k * n];
    }
    if (sum <= 0.0) {
      return -1;  /* Not positive definite */
    }
    A[j + j * n] = sqrt(sum);
    
    for (int i = j + 1; i < n; ++i) {
      sum = A[i + j * n];
      for (int k = 0; k < j; ++k) {
        sum -= A[i + k * n] * A[j + k * n];
      }
      A[i + j * n] = sum / A[j + j * n];
    }
  }
  return 0;
}

/* 
 * Simple Jacobi eigenvalue algorithm for symmetric matrices
 * A is n x n symmetric, stored in column-major
 * On exit: w contains eigenvalues (unsorted), V contains eigenvectors
 */
static void simple_jacobi_eigen(int n, real_type *A, real_type *w, real_type *V) {
  int max_iter = 100 * n * n;
  real_type eps = 1e-14;
  
  /* Initialize V to identity */
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      V[i + j * n] = (i == j) ? 1.0 : 0.0;
    }
  }
  
  /* Work on copy of A */
  real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
  for (int i = 0; i < n * n; ++i) {
    Acopy[i] = A[i];
  }
  
  for (int iter = 0; iter < max_iter; ++iter) {
    /* Find largest off-diagonal element */
    int p = 0, q = 1;
    real_type max_val = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        real_type absval = fabs(Acopy[i + j * n]);
        if (absval > max_val) {
          max_val = absval;
          p = i;
          q = j;
        }
      }
    }
    
    if (max_val < eps) break;
    
    /* Compute Jacobi rotation */
    real_type app = Acopy[p + p * n];
    real_type aqq = Acopy[q + q * n];
    real_type apq = Acopy[p + q * n];
    
    real_type theta = 0.5 * atan2(2.0 * apq, aqq - app);
    real_type c = cos(theta);
    real_type s = sin(theta);
    
    /* Apply rotation to Acopy */
    for (int i = 0; i < n; ++i) {
      if (i != p && i != q) {
        real_type aip = Acopy[i + p * n];
        real_type aiq = Acopy[i + q * n];
        Acopy[i + p * n] = c * aip - s * aiq;
        Acopy[p + i * n] = Acopy[i + p * n];
        Acopy[i + q * n] = s * aip + c * aiq;
        Acopy[q + i * n] = Acopy[i + q * n];
      }
    }
    Acopy[p + p * n] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    Acopy[q + q * n] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    Acopy[p + q * n] = 0.0;
    Acopy[q + p * n] = 0.0;
    
    /* Apply rotation to V */
    for (int i = 0; i < n; ++i) {
      real_type vip = V[i + p * n];
      real_type viq = V[i + q * n];
      V[i + p * n] = c * vip - s * viq;
      V[i + q * n] = s * vip + c * viq;
    }
  }
  
  /* Extract eigenvalues */
  for (int i = 0; i < n; ++i) {
    w[i] = Acopy[i + i * n];
  }
  
  /* Sort eigenvalues and eigenvectors in ascending order */
  for (int i = 0; i < n - 1; ++i) {
    int min_idx = i;
    for (int j = i + 1; j < n; ++j) {
      if (w[j] < w[min_idx]) min_idx = j;
    }
    if (min_idx != i) {
      /* Swap eigenvalues */
      real_type tmp = w[i];
      w[i] = w[min_idx];
      w[min_idx] = tmp;
      /* Swap eigenvector columns */
      for (int k = 0; k < n; ++k) {
        tmp = V[k + i * n];
        V[k + i * n] = V[k + min_idx * n];
        V[k + min_idx * n] = tmp;
      }
    }
  }
  
  free(Acopy);
}

/* Standard symmetric eigenvalue problem: A*x = lambda*x */
void simple_dsyev(const int n,
                  real_type *A,
                  real_type *w,
                  real_type *eigvecs) {
  simple_jacobi_eigen(n, A, w, eigvecs);
}

/* Generalized symmetric eigenvalue problem: A*x = lambda*B*x */
/* Uses Cholesky factorization: B = L*L', then solve L^{-1}*A*L^{-T}*y = lambda*y */
/* where x = L^{-T}*y */
void simple_dsygv(const int n,
                  real_type *A,
                  real_type *B,
                  real_type *w,
                  real_type *eigvecs) {
  
  /* Make copies since we'll modify */
  real_type *Bcopy = (real_type*) malloc(n * n * sizeof(real_type));
  real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
  real_type *C = (real_type*) malloc(n * n * sizeof(real_type));
  
  for (int i = 0; i < n * n; ++i) {
    Bcopy[i] = B[i];
    Acopy[i] = A[i];
  }
  
  /* Compute condition estimate based on diagonal range */
  real_type min_diag = fabs(B[0]);
  real_type max_diag = fabs(B[0]);
  real_type trace = 0.0;
  for (int i = 0; i < n; ++i) {
    real_type d = fabs(B[i + i * n]);
    trace += d;
    if (d < min_diag) min_diag = d;
    if (d > max_diag) max_diag = d;
  }
  
  /* Preemptive regularization if B appears ill-conditioned */
  real_type cond_est = (min_diag > 1e-14) ? max_diag / min_diag : 1e14;
  if (cond_est > 1e10 || min_diag < 1e-12) {
    real_type reg = 1e-10 * (trace / n + 1.0);
    if (min_diag < 1e-12) {
      reg = 1e-8 * (trace / n + 1.0);
    }
    for (int i = 0; i < n; ++i) {
      Bcopy[i + i * n] = B[i + i * n] + reg;
    }
  }
  
  /* Cholesky: B = L*L' */
  int ret = simple_cholesky(n, Bcopy);
  if (ret != 0) {
    /* Cholesky failed - use progressive regularization */
    real_type reg = 1e-8 * (trace / n + 1.0);
    for (int attempt = 0; attempt < 5 && ret != 0; ++attempt) {
      for (int i = 0; i < n; ++i) {
        Bcopy[i + i * n] = B[i + i * n] + reg;
      }
      ret = simple_cholesky(n, Bcopy);
      reg *= 10.0;
    }
    if (ret != 0) {
      for (int i = 0; i < n; ++i) {
        Bcopy[i + i * n] = B[i + i * n] + 1e-4 * (trace / n + 1.0);
      }
      simple_cholesky(n, Bcopy);
    }
  }
  
  /* L is now in lower triangle of Bcopy */
  /* Compute C = L^{-1} * A */
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      real_type sum = Acopy[i + j * n];
      for (int k = 0; k < i; ++k) {
        sum -= Bcopy[i + k * n] * C[k + j * n];
      }
      C[i + j * n] = sum / Bcopy[i + i * n];
    }
  }
  
  /* Compute Acopy = C * L^{-T} = L^{-1} * A * L^{-T} */
  for (int j = 0; j < n; ++j) {
    for (int i = n - 1; i >= 0; --i) {
      real_type sum = C[j + i * n];
      for (int k = i + 1; k < n; ++k) {
        sum -= Bcopy[k + i * n] * Acopy[j + k * n];
      }
      Acopy[j + i * n] = sum / Bcopy[i + i * n];
    }
  }
  
  /* Symmetrize */
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      real_type avg = 0.5 * (Acopy[i + j * n] + Acopy[j + i * n]);
      Acopy[i + j * n] = avg;
      Acopy[j + i * n] = avg;
    }
  }
  
  /* Solve standard eigenvalue problem */
  simple_jacobi_eigen(n, Acopy, w, eigvecs);
  
  /* Back-transform eigenvectors: x = L^{-T} * y */
  for (int k = 0; k < n; ++k) {
    for (int i = n - 1; i >= 0; --i) {
      real_type sum = eigvecs[i + k * n];
      for (int j = i + 1; j < n; ++j) {
        sum -= Bcopy[j + i * n] * eigvecs[j + k * n];
      }
      eigvecs[i + k * n] = sum / Bcopy[i + i * n];
    }
  }
  
  free(Bcopy);
  free(Acopy);
  free(C);
}

/* Vector 2-norm */
real_type simple_nrm2(const int n, const real_type *v) {
  real_type sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += v[i] * v[i];
  }
  return sqrt(sum);
}

/* Set vector elements to value */
void simple_vec_set(const int n, real_type value, real_type *vec) {
  for (int i = 0; i < n; ++i) {
    vec[i] = value;
  }
}
