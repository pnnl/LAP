#include "common.h"
#ifndef HIPBLAS_H
#define HIPBLAS_H

void initialize_handles();

void analyze_spmv(const int n, 
                  const int nnz, 
                  int *ia, 
                  int *ja, 
                  real_type *a, 
                  const real_type *x, 
                  real_type *result,
                 char * option);

void initialize_and_analyze_L_and_U_solve(const int n, 
                                              const int nnzL, 
                                              int *lia, 
                                              int *lja, 
                                              real_type *la,
                                              const int nnzU, 
                                              int *uia, 
                                              int *uja, 
                                              real_type *ua);

real_type hip_dot (const int n, const real_type *v, const real_type *w);

void hip_scal (const int n, const real_type alpha, real_type *v);

void hip_axpy (const int n, const real_type alpha, const real_type *x, real_type *y);

void hip_csr_matvec(const int n, 
                    const int nnz, 
                    const int *ia, 
                    const int *ja, 
                    const real_type *a, 
                    const real_type *x, 
                    real_type *result, 
                    const real_type *al, 
                    const real_type *bet, 
                    const char *kind);

void hip_lower_triangular_solve(const int n, 
                                const int nnzL, 
                                const int *lia, 
                                const int *lja, 
                                const real_type *la,
                                const real_type *diagonal, 
                                const real_type *x, 
                                real_type *result);

void hip_upper_triangular_solve(const int n,
                                const int nnzU, 
                                const int *uia, 
                                const int *uja, 
                                const real_type *ua, 
                                const real_type *diagonal, 
                                const real_type *x, 
                                real_type *result);

void hip_vec_vec(const int n, const real_type *x, const real_type *y, real_type *res);

void hip_vector_reciprocal(const int n, const real_type *v, real_type *res);

void hip_vector_sqrt(const int n, const real_type *v, real_type *res);

void hip_vec_copy(const int n, const real_type *src, real_type *dest);

void hip_vec_zero(const int n, real_type *vec);

void hip_vec_set(const int n, real_type value, real_type *vec);

void initialize_ichol(const int n, 
                      const int nnzA, 
                      int *ia, 
                      int *ja, 
                      real_type *a);

void hip_ichol(const int *ia, 
               const int *ja, 
               real_type *a, 
               const int nnzA, 
               pdata *prec_data, 
               real_type *x, 
               real_type *y);
#endif
