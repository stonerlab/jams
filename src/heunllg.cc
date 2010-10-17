#include "globals.h"

#include "heunllg.h"
#include "array2d.h"
#include <cmath>
// #include "cblas.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


void HeunLLGSolver::initialise(int argc, char **argv, double idt)
{
  
  // initialise base class
  Solver::initialise(argc,argv,idt);

  output.write("Initialising Heun LLG solver (CPU)\n");

  snew.resize(globals::nspins,3);

  initialised = true;
}

void HeunLLGSolver::run()
{
  using namespace globals;
  // Calculate fields
  
  // exchange and anisotropy
#ifdef MKL
  const char transa = "N"
  const char matdescra[6] = "SUNC";
  const double one = 1.0;
  mkl_dcsrmv(transa,&nspins,&nspins,&one,matdescra,val,col,
      jijxx.ptrB(),jijxx.ptrE(),&(s[0]),&one,&(h[0]));
#endif

  for(int i=0; i<nspins; ++i) {
    double sxh[3] =
      { s(i,1)*h(i,2) - s(i,2)*h(i,1),
        s(i,2)*h(i,0) - s(i,0)*h(i,2),
        s(i,0)*h(i,1) - s(i,1)*h(i,0) };

    double rhs[3] =
      { sxh[0] + alpha(i) * ( s(i,1)*sxh[2] - s(i,2)*sxh[1] ),
        sxh[1] + alpha(i) * ( s(i,2)*sxh[0] - s(i,0)*sxh[2] ),
        sxh[2] + alpha(i) * ( s(i,0)*sxh[1] - s(i,1)*sxh[0] ) };

    for(int n=0; n<3; ++n) {
      snew(i,n) = s(i,n) + 0.5*dt*rhs[n];
    }

    for(int n=0; n<3; ++n) {
      s(i,n) = s(i,n) + dt*rhs[n];
    }

    double norm = 1.0/sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));

    for(int n=0; n<3; ++n) {
      s(i,n) = s(i,n)*norm;
    }
  }

  // Calculate fields

  for(int i=0; i<nspins; ++i)
  {
    double sxh[3] =
      { s(i,1)*h(i,2) - s(i,2)*h(i,1),
        s(i,2)*h(i,0) - s(i,0)*h(i,2),
        s(i,0)*h(i,1) - s(i,1)*h(i,0) };
    
    double rhs[3] =
      { sxh[0] + alpha(i) * ( s(i,1)*sxh[2] - s(i,2)*sxh[1] ),
        sxh[1] + alpha(i) * ( s(i,2)*sxh[0] - s(i,0)*sxh[2] ),
        sxh[2] + alpha(i) * ( s(i,0)*sxh[1] - s(i,1)*sxh[0] ) };

    for(int n=0; n<3; ++n) {
      s(i,n) = snew(i,n) + 0.5*dt*rhs[n];
    }
    
    double norm = 1.0/sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));

    for(int n=0; n<3; ++n) {
      s(i,n) = s(i,n)*norm;
    }
  }
}
