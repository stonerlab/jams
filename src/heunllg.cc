#include "globals.h"

#include "heunllg.h"
#include "vecfield.h"
// #include "cblas.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif

HeunLLGSolver::HeunLLGSolver(){

}

HeunLLGSolver::~HeunLLGSolver(){

}

void HeunLLGSolver::initialise(int argc, char **argv, double dt)
{
  
  // initialise base class
  Solver::initialise(argc,argv,dt);

  output.write("Initialising Heun LLG solver (CPU)\n");

  snew.resize(globals::nspins);

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
      { s.y(i)*h.z(i) - s.z(i)*h.y(i),
        s.z(i)*h.x(i) - s.x(i)*h.z(i),
        s.x(i)*h.y(i) - s.y(i)*h.x(i) };

    double rhs[3] =
      { sxh[0] + alpha(i) * ( s.y(i)*sxh[2] - s.z(i)*sxh[1] ),
        sxh[1] + alpha(i) * ( s.z(i)*sxh[0] - s.x(i)*sxh[2] ),
        sxh[2] + alpha(i) * ( s.x(i)*sxh[1] - s.y(i)*sxh[0] ) };

    snew.x(i) = s.x(i) + 0.5*dt*rhs[0];
    snew.y(i) = s.y(i) + 0.5*dt*rhs[1];
    snew.z(i) = s.z(i) + 0.5*dt*rhs[2];

    s.x(i) = s.x(i) + dt*rhs[0];
    s.y(i) = s.y(i) + dt*rhs[1];
    s.z(i) = s.z(i) + dt*rhs[2];

    double norm = 1.0/sqrt(s.x(i)*s.x(i) + s.y(i)*s.y(i) +
      s.z(i)*s.z(i));

    s.x(i) = s.x(i)*norm;
    s.y(i) = s.y(i)*norm;
    s.z(i) = s.z(i)*norm;
  }

  // Calculate fields

  for(int i=0; i<nspins; ++i)
  {
    double sxh[3] =
      { s.y(i)*h.z(i) - s.z(i)*h.y(i),
        s.z(i)*h.x(i) - s.x(i)*h.z(i),
        s.x(i)*h.y(i) - s.y(i)*h.x(i) };

    s.x(i) = snew.x(i) + 0.5*dt*(sxh[0] + alpha(i) *
      ( s.y(i)*sxh[2] - s.z(i)*sxh[1]) );
    s.y(i) = snew.y(i) + 0.5*dt*(sxh[1] + alpha(i) *
      ( s.y(i)*sxh[0] - s.z(i)*sxh[2]) );
    s.z(i) = snew.z(i) + 0.5*dt*(sxh[2] + alpha(i) * 
      ( s.y(i)*sxh[1] - s.z(i)*sxh[0]) );

    double norm = 1.0/sqrt(s.x(i)*s.x(i) + s.y(i)*s.y(i) +
      s.z(i)*s.z(i));

    s.x(i) = s.x(i)*norm;
    s.y(i) = s.y(i)*norm;
    s.z(i) = s.z(i)*norm;
  }
}
