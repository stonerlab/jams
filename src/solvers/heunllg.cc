#include "globals.h"
#include "consts.h"
#include "fields.h"

#include "heunllg.h"
#include "array2d.h"
#include <cmath>
//#include "cblas.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


void HeunLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;
  
  // initialise base class
  Solver::initialise(argc,argv,idt);

  output.write("Initialising Heun LLG solver (CPU)\n");

  output.write("  * Converting MAP to CSR\n");
  Jij.convertMAP2CSR();
  J2ij.convertMAP2CSR();
  output.write("  * Jij matrix memory (CSR): %f MB\n",Jij.calculateMemory());
  output.write("  * J2ij matrix memory (CSR): %f MB\n",J2ij.calculateMemory());

  snew.resize(nspins,3);
  sigma.resize(nspins,3);
  
  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)*mu_bohr_si) );
    }
  }

  initialised = true;
}

void HeunLLGSolver::syncOutput()
{

}

void HeunLLGSolver::run()
{
  using namespace globals;

  int i,j;
  double sxh[3], rhs[3];
  double norm;


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

  iteration++;
}
