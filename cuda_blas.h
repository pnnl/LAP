
#include "common.h"
#ifndef CUDABLAS_H
#define CUDABLAS_H

real_type cuda_dot (const int n, const real_type *v, const real_type *w);

void cuda_scal (const int n, const real_type alpha, real_type *v);

void cuda_axpy (const int n, const real_type alpha, const real_type *x, real_type *y);

void cuda_csr_matvec(const int n, 
                     const int nnz, 
                     const int *ia, 
                     const int *ja, 
                     const real_type *a, 
                     const real_type *x, 
                     real_type *result, 
                     const  real_type *al, 
                     const real_type *bet);

void cuda_lower_triangular_solve(const int n,
                                 const int nnz,
                                 const int *lia,
                                 const int *lja,
                                 const real_type *la,
                                 const real_type *diag, 
                                 const real_type *x,
                                 real_type *result);

void cuda_upper_triangular_solve(const int n, 
                                 const int nnz, 
                                 const int *uia, 
                                 const int *uja, 
                                 const real_type *ua, 
                                 const real_type *diag,
                                 const real_type *x, 
                                 real_type *result);

void cuda_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res);

void cuda_vector_sqrt(const int n, const real_type *v, real_type *res);

void cuda_vector_reciprocal(const int n, const real_type *v, real_type *res);

void cuda_vec_copy(const int n, const real_type *src, real_type *dest);

void cuda_vec_zero(const int n, real_type *vec);

void initialize_handles();

void initialize_spmv_buffer(const int n, 
                            const int nnz, 
                            int *ia, 
                            int *ja, 
                            real_type *a, 
                            const real_type *x, 
                            real_type *result, 
                            real_type *al, 
                            real_type *bet);


void initialize_and_analyze_L_and_U_solve(const int n, 
                                          const int nnzL, 
                                          int *lia, 
                                          int *lja, 
                                          real_type *la,
                                          const int nnzU, 
                                          int *uia, 
                                          int *uja, 
                                          real_type *ua);

void initialize_L_and_U_descriptors(const int n, 
                                  const int nnzL, 
                                  int *lia, 
                                  int *lja, 
                                  real_type *la,
                                  const int nnzU, 
                                  int *uia, 
                                  int *uja, 
                                  real_type *ua);

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a);

void cuda_ichol(const int *ia, 
                const int *ja, 
                real_type *a, 
                const int nnzA, 
                pdata *prec_data, 
                real_type *x, 
                real_type *y);
#endif
