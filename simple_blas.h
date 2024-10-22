//prototypes
//
#include "common.h"
#ifndef SIMPLEBLAS_H
#define SIMPLEBLAS_H
real_type simple_dot (const int n, const real_type *v, const real_type *w);

void simple_scal (const int n, const real_type alpha, real_type *v);

void simple_axpy (const int n, const real_type alpha, const real_type *x, real_type *y);

void simple_csr_matvec(const int n, 
                       const int nnz, 
                       const int *ia, 
                       const int *ja, 
                       const real_type *a, 
                       const real_type *x, 
                       real_type *result, 
                       const  real_type *al, 
                       const real_type *bet);

void simple_lower_triangular_solve(const int n, 
                                   const int nnz,
                                   const int *lia,
                                   const int *lja, 
                                   const real_type *la,
                                   const real_type *diag, 
                                   const real_type *x, 
                                   real_type *result);

void simple_upper_triangular_solve(const int n,
                                   const int nnz, 
                                   const int *uia, 
                                   const int *uja, 
                                   const real_type *ua, 
                                   const real_type *diag,
                                   const real_type *x, 
                                   real_type *result);

void simple_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res);

void simple_vector_sqrt(const int n, const real_type *v, real_type *res);

void simple_vector_reciprocal(const int n, const real_type *v, real_type *res);

void simple_vec_copy(const int n, const real_type *src, real_type *dest);

void simple_vec_zero(const int n, real_type *vec);

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a, 
                      int *lia,
                      int *lja,
                      real_type *la);

void simple_ichol(const int *ia, const int *ja, real_type *a, int nnzA, pdata *prec_data, real_type *x, real_type *y);
#endif
