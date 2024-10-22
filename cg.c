#include "common.h"
#include "blas.h"
#include "devMem.h"

void cg(int n, real_type nnz,
        int *ia, //matrix csr data
        int *ja,
        real_type *a,
        real_type *x, //solution vector, mmust be alocated prior to calling
        real_type *b, //rhs
        real_type tol, //DONT MULTIPLY BY NORM OF B
        pdata* prec_data, //preconditioner data: all Ls, Us etc
        int maxit,
        int *it, //output: iteration
        int *flag, //output: flag 0-converged, 1-maxit reached, 2-catastrophic failure
        real_type *res_norm_history //output: residual norm history
       ){

#if (CUDA || HIP)
  real_type *r;
  real_type *w;
  real_type *p;
  real_type *q;

  r = (real_type*) mallocForDevice(r, n, sizeof(real_type));
  w = (real_type*) mallocForDevice(w, n, sizeof(real_type));
  p = (real_type*) mallocForDevice(p, n, sizeof(real_type));
  q = (real_type*) mallocForDevice(q, n, sizeof(real_type));
#if HIP
  vec_zero(n, w);
  vec_zero(n, r);
  vec_zero(n, p);
  vec_zero(n, q);
#endif
#else
  real_type *r = (real_type *) calloc (n, sizeof(real_type));
  real_type *w = (real_type *) calloc (n, sizeof(real_type));
  real_type *p = (real_type *) calloc (n, sizeof(real_type));
  real_type *q = (real_type *) calloc (n, sizeof(real_type));
#endif

  real_type alpha, beta, tolrel, rho_current, rho_previous, pTq;
  real_type one = 1.0;
  real_type zero = 0.0;  
  int notconv = 1, iter = 0;
  //compute initial norm of r
  /* r = A*x */
  csr_matvec(n, nnz, ia, ja, a, x, r, &one, &zero, "A");

  /* r = -b +r = Ax-b */
  axpy(n, -1.0, b, r);
  // printf("Norm of r %e \n", dot(n, r,r));  

  /* r=(-1.0)*r */
  scal(n, -1.0, r);

  /* norm of r */
  res_norm_history[0] = dot(n, r,r);
  res_norm_history[0] = sqrt(res_norm_history[0]);
  tolrel = tol * res_norm_history[0];

  printf("CG: it %d, res norm %5.5e \n",0, res_norm_history[0]);

  while (notconv){
    // printf("Norm of X before prec %16.16e \n", dot(n, r,r));  
#if HIP
    vec_zero(n, w);
#endif
    prec_function(ia, ja, a, nnz, prec_data, r, w);
    // printf("Norm of X after prec %16.16e \n", dot(n, w,w));  

    /* rho_current = r'*w; */
    rho_current = dot(n, r, w);
    if (iter == 0) {
      vec_copy(n, w, p);
    } else {
      beta = rho_current/rho_previous;
      // printf("scaling by beta = %5.5e, rho_current = %5.5e, rho_previous = %5.5e \n", beta, rho_current, rho_previous);
      /* p = w+bet*p; */

      scal(n, beta, p);
      axpy(n, 1.0, w, p);
    }
    /* q = As*p; */
    csr_matvec(n, nnz, ia, ja, a, p, q, &one, &zero, "A");

    /* alpha = rho_current/(p'*q);*/
    // printf("p'*p = %16.16e, q'*q = %16.16f \n",dot(n, p,p), dot(n, q,q) );

    pTq = dot(n, p, q);
    alpha = rho_current / pTq; 
    // printf("p^Tq = %5.15e,rho_current = %5.15e, alpha = %5.15e \n", pTq, rho_current, alpha);

    /* x = x + alph*p; */
    axpy(n, alpha, p, x);

    /* r = r - alph*q; */
    axpy(n, (-1.0) * alpha, q, r );

    /* norm of r */
    iter++;
    res_norm_history[iter] = dot(n, r,r);
    res_norm_history[iter] = sqrt(res_norm_history[iter]);
    printf("CG: it %d, res norm %5.5e \n",iter, res_norm_history[iter]);

    /* check convergence */
    if ((res_norm_history[iter]) < tolrel) {
      *flag = 0;
      notconv = 0;
      *it = iter; 
  
      /* r = A*x */
      csr_matvec(n, nnz, ia, ja, a, x, r, &one, &zero, "A");

      /* r = -b +r = Ax-b */
      axpy(n, -1.0, b, r);
      printf("TRUE Norm of r %5.5e relative %16.16e\n", sqrt(dot(n, r,r)), sqrt(dot(n, r,r))/sqrt(dot(n, b,b)));  
  } else {
      if (iter > maxit){
        *flag = 1;
        notconv = 0;
        *it = iter; 
      }
    }
    rho_previous = rho_current;
  }//while


#if (CUDA || HIP)
  freeDevice(r);
  freeDevice(w);
  freeDevice(p);
  freeDevice(q);
#else
  free(r);
  free(w);
  free(p);
  free(q);
#endif
} /* cg */

