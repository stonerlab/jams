#include "globals.h"
#include "consts.h"
#include "fields.h"

#include "semillg.h"
#include "array2d.h"
#include <cmath>
#include <algorithm>
//#include "cblas.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


void SemiLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;
  
  // initialise base class
  Solver::initialise(argc,argv,idt);

  output.write("Initialising Semi Implicit LLG solver (CPU)\n");

  sold.resize(nspins,3);
  sigma.resize(nspins,3);
    
  temperature = 300.0;

  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*mus(i)) / (dt) );
    }
  }

  initialised = true;
}

void SemiLLGSolver::run()
{
  using namespace globals;

  int i,j;
  double sxh[3], fxs[3], f[3];
  double norm;
  double b2ff,fdots;

  // copy s to sold
  std::copy(s.ptr(),s.ptr()+nspins3,sold.ptr());
  
  if(temperature > 0.0) {
    const double stmp = sqrt(temperature);
    for(i=0; i<nspins; ++i) {
      for(j=0; j<3; ++j) {
        w(i,j) = (rng.normal())*sigma(i,j)*stmp;
      }
    }
  }

  calculate_fields();
  
  for(i=0; i<nspins; ++i) {
    
    sxh[0] = s(i,1)*h(i,2) - s(i,2)*h(i,1);
    sxh[1] = s(i,2)*h(i,0) - s(i,0)*h(i,2);
    sxh[2] = s(i,0)*h(i,1) - s(i,1)*h(i,0);
    
    for(j=0; j<3; ++j){
      f[j] = -0.5*dt*( h(i,j) + alpha(i)*sxh[j] );
    }

    b2ff = (f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);

    norm = 1.0/(1.0+b2ff);

    fdots = (f[0]*s(i,0)+f[1]*s(i,1)+f[2]*s(i,2));

    fxs[0] = (f[1]*s(i,2) - f[2]*s(i,1));  
    fxs[1] = (f[2]*s(i,0) - f[0]*s(i,2));
    fxs[2] = (f[0]*s(i,1) - f[1]*s(i,0));
    
    for(j=0;j<3;++j) {
      s(i,j) = 0.5 * ( s(i,j) + ( s(i,j)*(1.0-b2ff) + 2.0*(fxs[j]+f[j]*fdots) )*norm );
    }

  }
  

  calculate_fields();

  for(i=0; i<nspins; ++i) {

    sxh[0] = s(i,1)*h(i,2) - s(i,2)*h(i,1);
    sxh[1] = s(i,2)*h(i,0) - s(i,0)*h(i,2);
    sxh[2] = s(i,0)*h(i,1) - s(i,1)*h(i,0);
    
    for(j=0; j<3; ++j){
      f[j] = -0.5*dt*( h(i,j) + alpha(i)*sxh[j] );
    }

    b2ff = (f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);

    norm = 1.0/(1.0+b2ff);

    fdots = (f[0]*sold(i,0)+f[1]*sold(i,1)+f[2]*sold(i,2));

    fxs[0] = (f[1]*sold(i,2) - f[2]*sold(i,1));  
    fxs[1] = (f[2]*sold(i,0) - f[0]*sold(i,2));
    fxs[2] = (f[0]*sold(i,1) - f[1]*sold(i,0));

    for(j=0;j<3;++j) {
      s(i,j) = norm*(sold(i,j)*(1.0-b2ff) + 2.0*(fxs[j]+f[j]*fdots));
    }
  }
}
