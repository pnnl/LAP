#include "openmp_blas.h"
#include <math.h>

void openmp_scal (const int n, const real_type alpha, real_type *v)
{ 
  int i;
#pragma omp target simd  map(alpha) map(tofrom:v[0:n])
  for (i = 0; i < n; ++i) {
    v[i] *= alpha;
  }
}

void openmp_axpy (const int n, const real_type alpha, const real_type *x, real_type *y){
  int i;
#pragma omp target simd  map(to:x[0:n]) map(tofrom:y[0:n])
  for (i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

void openmp_csr_matvec(const int n, 
                       const int nnz, 
                       const int *ia, 
                       const int *ja, 
                       const real_type *a, 
                       const real_type *x, 
                       real_type *result,
                       const  real_type *al,
                       const real_type *bet) 
{

  real_type alpha = *al;
  real_type beta = *bet;
  /* go through every row */
  int i, j, col;
  real_type s;
#pragma omp parallel num_threads(16)
{
//#pragma omp target teams distribute parallel for   private(s, j) map(to:a[0:nnz], x[0:n], ia[0:n+1], ja[0:nnz], alpha, beta) map(tofrom:result[0:n])
 #pragma omp for private(s,j) schedule(static) 
 for (i = 0; i < n; ++i) {
    /* go through each column in this row */
    s = result[i] * beta;  
#pragma omp reduction(+:s)
    for (j = ia[i]; j < ia[i + 1]; j++) {
      col = ja[j];
      s += (alpha * a[j] * x[col]);
    }
    result[i] = s;
  }
}
}

void openmp_lower_triangular_solve(const int n, 
                                   const int nnz, 
                                   const int *lia, 
                                   const int *lja, 
                                   const real_type *la,
                                   const real_type *diagonal, 
                                   const real_type *x, 
                                   real_type *result) 
{
  /* compute result = L^{-1}x */ 
  int i, j, col;
  //#pragma omp target teams distribute map(to:lia[0:n+1],lja[0:nnz],la[0:nnz],x[0:n], diagonal[0:n])  map(tofrom:result[0:n])
  for (i = 0; i < n; ++i) {
    real_type s = 0.0;
    #pragma omp simd private(col) reduction(+:s)
    for (j = lia[i]; j < lia[i + 1]; ++j) {
      col = lja[j];
      s += (-1.0) * la[j] * result[col]; 
    }

    result[i] = (s + x[i]) / diagonal[i];
  }
}

void openmp_upper_triangular_solve(const int n, 
                                   const int nnz, 
                                   const int *uia, 
                                   const int *uja, 
                                   const real_type *ua, 
                                   const real_type *diagonal, 
                                   const real_type *x, 
                                   real_type *result)
{
  /* compute result = U^{-1}x */ 
  /* go through each row (starting from the last row) */
  int i, j, col;
  real_type s; 
  //this kind of works but the result is non deterministic 
//#pragma omp target teams distribute map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  // #pragma omp target map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz], x[0:n], diagonal[0:n]) map(tofrom:result[0:n]) 
  for (i = n - 1; i >= 0; --i) {
    s = 0.0;
    result[i] = 0.0;
#pragma omp simd private( col) reduction(+:s)
    //map(to:uia[0:n+1],uja[0:nnz],ua[0:nnz],result[0:n]) map(from:s)
    for (j = uia[i]; j < uia[i + 1]; ++j){
      col = uja[j];
      s += (-1.0) * ua[j] * result[col];
    }
    //#pragma omp ordered
    result[i] = (s + x[i]) / diagonal[i]; 
  }
}

/* not std blas but needed and embarassingly parallel */

/* simple vec-vec computes an element-wise product (needed for scaling) */
void openmp_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res)
{
  int i;
#pragma omp target simd  map(to:x[0:n], y[0:n]) map(from:res[0:n])
  for (i = 0; i < n; ++i) {
    res[i] = x[i] * y[i];
  }
}

/* vector reciprocal computes 1./d */
void openmp_vector_reciprocal(const int n, const real_type *v, real_type *res)
{
  int i;
#pragma omp target simd  map(to:v[0:n]) map(from:res[0:n])
  for (i = 0; i < n; ++i) {
    if (v[i] != 0.0) {
      res[i] = 1.0 / v[i];
    } else { 
      res[i] = 0.0;
    }
  }
}

/* vector sqrt takes an sqrt from each vector entry */
void openmp_vector_sqrt(const int n, const real_type *v, real_type *res)
{
  int i;
#pragma omp target simd   map(to:v[0:n]) map(from:res[0:n])
  for (i = 0; i < n; ++i) {
    if  (v[i] >= 0.0) {
      res[i] = sqrt(v[i]);
    } else {
      res[i] = 0.0;
    }
  }
}

void openmp_vec_copy(const int n, const real_type *src, real_type *dest)
{
  int i;
#pragma omp target simd  map(to:src[0:n]) map(from:dest[0:n])
  for (i = 0; i < n; ++i) {
    dest[i] = src[i];  
  }
}

void openmp_vec_zero(const int n, real_type *vec)
{
  int i;
#pragma omp target simd    map(tofrom:vec[0:n])
  for (i = 0; i < n; ++i) {
    vec[i] = 0.0;  
  }
}

void openmp_gemv(const char *T,
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
  int i, j;
  double a = *alpha;
  double b = *beta;
  
  if (T[0] == 'T' || T[0] == 't') {
    /* Transpose case: y = alpha * A^T * x + beta * y */
    /* A is m x n, A^T is n x m, x is m-vector, y is n-vector */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < n; ++j) {
      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (i = 0; i < m; ++i) {
        sum += A[i + j * lda] * x[i];
      }
      y[j] = a * sum + b * y[j];
    }
  } else {
    /* Non-transpose case: y = alpha * A * x + beta * y */
    /* A is m x n, x is n-vector, y is m-vector */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < m; ++i) {
      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (j = 0; j < n; ++j) {
        sum += A[i + j * lda] * x[j];
      }
      y[i] = a * sum + b * y[i];
    }
  }
}

real_type openmp_dot(const int n, const real_type *v, const real_type *w)
{
  real_type sum = 0.0;
  int i;
//#pragma omp target teams distribute parallel for    map(to:v[0:n], w[0:n]) reduction(+:sum)  
#pragma omp parallel for reduction(+:sum)
for (i = 0; i < n; ++i) {
    sum += (v[i] * w[i]);
  }
  return sum;
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

void openmp_ichol(const int *ia, 
                  const int *ja, 
                  real_type *a, 
                  const int nnzA, 
                  pdata *prec_data, 
                  real_type *x, 
                  real_type *y)
{
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
void openmp_gemm(const char *transA,
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
  int i, j, l;
  
  #pragma omp parallel for private(i, l) schedule(static)
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      real_type sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (l = 0; l < k; ++l) {
        real_type aval = ta ? A[l + i * lda] : A[i + l * lda];
        real_type bval = tb ? B[j + l * ldb] : B[l + j * ldb];
        sum += aval * bval;
      }
      C[i + j * ldc] = a * sum + b * C[i + j * ldc];
    }
  }
}

/* Vector 2-norm */
real_type openmp_nrm2(const int n, const real_type *v) {
  real_type sum = 0.0;
  int i;
  #pragma omp parallel for reduction(+:sum)
  for (i = 0; i < n; ++i) {
    sum += v[i] * v[i];
  }
  return sqrt(sum);
}

/* Set vector elements to value */
void openmp_vec_set(const int n, real_type value, real_type *vec) {
  int i;
  #pragma omp parallel for
  for (i = 0; i < n; ++i) {
    vec[i] = value;
  }
}

/* 
 * Jacobi eigenvalue algorithm for symmetric matrices
 */
static void openmp_jacobi_eigen(int n, real_type *A, real_type *w, real_type *V) {
  int max_iter = 100 * n * n;
  real_type eps = 1e-14;
  
  /* Initialize V to identity */
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      V[i + j * n] = (i == j) ? 1.0 : 0.0;
    }
  }
  
  real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
  for (int i = 0; i < n * n; ++i) {
    Acopy[i] = A[i];
  }
  
  for (int iter = 0; iter < max_iter; ++iter) {
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
    
    real_type app = Acopy[p + p * n];
    real_type aqq = Acopy[q + q * n];
    real_type apq = Acopy[p + q * n];
    
    real_type theta = 0.5 * atan2(2.0 * apq, aqq - app);
    real_type c = cos(theta);
    real_type s = sin(theta);
    
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
    
    for (int i = 0; i < n; ++i) {
      real_type vip = V[i + p * n];
      real_type viq = V[i + q * n];
      V[i + p * n] = c * vip - s * viq;
      V[i + q * n] = s * vip + c * viq;
    }
  }
  
  for (int i = 0; i < n; ++i) {
    w[i] = Acopy[i + i * n];
  }
  
  /* Sort */
  for (int i = 0; i < n - 1; ++i) {
    int min_idx = i;
    for (int j = i + 1; j < n; ++j) {
      if (w[j] < w[min_idx]) min_idx = j;
    }
    if (min_idx != i) {
      real_type tmp = w[i];
      w[i] = w[min_idx];
      w[min_idx] = tmp;
      for (int k = 0; k < n; ++k) {
        tmp = V[k + i * n];
        V[k + i * n] = V[k + min_idx * n];
        V[k + min_idx * n] = tmp;
      }
    }
  }
  
  free(Acopy);
}

static int openmp_cholesky(int n, real_type *A) {
  for (int j = 0; j < n; ++j) {
    real_type sum = A[j + j * n];
    for (int k = 0; k < j; ++k) {
      sum -= A[j + k * n] * A[j + k * n];
    }
    if (sum <= 0.0) return -1;
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

/* Standard symmetric eigenvalue problem */
void openmp_dsyev(const int n,
                  real_type *A,
                  real_type *w,
                  real_type *eigvecs) {
  openmp_jacobi_eigen(n, A, w, eigvecs);
}

/* Generalized symmetric eigenvalue problem: A*x = lambda*B*x */
void openmp_dsygv(const int n,
                  real_type *A,
                  real_type *B,
                  real_type *w,
                  real_type *eigvecs) {
  
  real_type *Bcopy = (real_type*) malloc(n * n * sizeof(real_type));
  real_type *Acopy = (real_type*) malloc(n * n * sizeof(real_type));
  real_type *C = (real_type*) malloc(n * n * sizeof(real_type));
  
  for (int i = 0; i < n * n; ++i) {
    Bcopy[i] = B[i];
    Acopy[i] = A[i];
  }
  
  int ret = openmp_cholesky(n, Bcopy);
  if (ret != 0) {
    fprintf(stderr, "Warning: Cholesky failed in openmp_dsygv. Using regularization.\n");
    for (int i = 0; i < n; ++i) {
      Bcopy[i + i * n] = B[i + i * n] + 1e-10;
    }
    openmp_cholesky(n, Bcopy);
  }
  
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
  
  /* Compute Acopy = C * L^{-T} */
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
  
  openmp_jacobi_eigen(n, Acopy, w, eigvecs);
  
  /* Back-transform eigenvectors */
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
