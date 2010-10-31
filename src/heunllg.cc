#include "globals.h"
#include "consts.h"
#include "fields.h"
#include "noise.h"

#include "heunllg.h"
#include "array2d.h"
#include <cmath>
//#include "cblas.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


void HeunLLGSolver::initialise(int argc, char **argv, double idt, NoiseType ntype)
{
  using namespace globals;
  
  // initialise base class
  Solver::initialise(argc,argv,idt,ntype);

  output.write("Initialising Heun LLG solver (CPU)\n");

  snew.resize(globals::nspins,3);

  initialised = true;
}

void HeunLLGSolver::run()
{
  using namespace globals;

  int i,j;
  double sxh[3], rhs[3];
  double norm;


  noise->run();

  calculate_fields();
  
  for(i=0; i<nspins; ++i) {
    
    sxh[0] = s(i,1)*h(i,2) - s(i,2)*h(i,1);
    sxh[1] = s(i,2)*h(i,0) - s(i,0)*h(i,2);
    sxh[2] = s(i,0)*h(i,1) - s(i,1)*h(i,0);

    rhs[0] = sxh[0] + alpha(i) * ( s(i,1)*sxh[2] - s(i,2)*sxh[1] );
    rhs[1] = sxh[1] + alpha(i) * ( s(i,2)*sxh[0] - s(i,0)*sxh[2] );
    rhs[2] = sxh[2] + alpha(i) * ( s(i,0)*sxh[1] - s(i,1)*sxh[0] );

    for(j=0; j<3; ++j) {
      snew(i,j) = s(i,j) + 0.5*dt*rhs[j];
    }

    for(j=0; j<3; ++j) {
      s(i,j) = s(i,j) + dt*rhs[j];
    }

    norm = 1.0/sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));

    for(j=0; j<3; ++j) {
      s(i,j) = s(i,j)*norm;
    }
  }
  
  noise->run();

  calculate_fields();

  for(i=0; i<nspins; ++i) {


    sxh[0] = s(i,1)*h(i,2) - s(i,2)*h(i,1);
    sxh[1] = s(i,2)*h(i,0) - s(i,0)*h(i,2);
    sxh[2] = s(i,0)*h(i,1) - s(i,1)*h(i,0);

    rhs[0] = sxh[0] + alpha(i) * ( s(i,1)*sxh[2] - s(i,2)*sxh[1] );
    rhs[1] = sxh[1] + alpha(i) * ( s(i,2)*sxh[0] - s(i,0)*sxh[2] );
    rhs[2] = sxh[2] + alpha(i) * ( s(i,0)*sxh[1] - s(i,1)*sxh[0] );

    for(j=0; j<3; ++j) {
      s(i,j) = snew(i,j) + 0.5*dt*rhs[j];
    }
    
    norm = 1.0/sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));

    for(j=0; j<3; ++j) {
      s(i,j) = s(i,j)*norm;
    }
  }
}
