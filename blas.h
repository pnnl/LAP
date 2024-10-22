#pragma once

real_type dot(const int n, const real_type *v, const real_type *w);

void axpy(const int n, const real_type alpha, real_type *x, real_type *y);

void scal(const int n, const real_type alpha, real_type *v);

void csr_matvec(const int n, 
                const int nnz, 
                const int *ia, 
                const int *ja, 
                const real_type *a, 
                const real_type *x, 
                real_type *result, 
                const  real_type *al, 
                const real_type *bet,
                const char *kind);

void lower_triangular_solve(const int n, 
                            const int nnz, 
                            const int *lia, 
                            const int *lja, 
                            const real_type *la,
                            const real_type *diag, 
                            const real_type *x, 
                            real_type *result);

void upper_triangular_solve(const int n, 
                            const int nnz, 
                            const int *uia, 
                            const int *uja, 
                            const real_type *ua,
                            const real_type *diag, 
                            const real_type *x, 
                            real_type *result);

void ichol(const int *ia, const int *ja, real_type *a, const int nnzA, pdata *prec_data, real_type *x, real_type *y);

void vec_vec(const int n, const real_type *x, real_type *y, real_type *res);

void vector_reciprocal(const int n, const real_type *v, real_type *res);

void vector_sqrt(const int n, const real_type *v, real_type *res);

void vec_copy(const int n, real_type *src, real_type *dest);

void vec_zero(const int n, real_type *vec);
