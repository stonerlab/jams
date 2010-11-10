#include "globals.h"
#include "consts.h"
#include "fields.h"

#include "heunllms.h"
#include "array2d.h"
#include <cmath>


void HeunLLMSSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;
  
  // initialise base class
  Solver::initialise(argc,argv,idt);

  output.write("Initialising Heun LLMS solver (CPU)\n");

  snew.resize(nspins,3);
  wnew.resize(nspins,3);
  u.resize(nspins,3);
  sigma.resize(nspins,3);
  
  omega_corr = 1.0/(10*dt);

  for(int i=0; i<nspins; ++i) {
    // restore gyro to just gamma
    gyro(i) = -gyro(i)*(1+alpha(i)*alpha(i));
    for(int j=0; j<3; ++j) {
      //sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*mus(i)) / (dt) );
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*omega_corr*omega_corr) / (dt*mus(i)) );
      w(i,j) = 0.0;
    }
  }

  initialised = true;
}

void HeunLLMSSolver::run()
{
  using namespace globals;
  assert(initialised);

  int i,j;
  double sxh[3], rhs[3];
  double norm;


  if(temperature > 0.0) {
    const double stmp = sqrt(temperature);
    for(i=0; i<nspins; ++i) {
      for(j=0; j<3; ++j) {
        u(i,j) = (rng.normal())*sigma(i,j)*stmp;
      }
    }
  }

  calculate_fields();
  
  for(i=0; i<nspins; ++i) {
    
    sxh[0] = s(i,1)*h(i,2) - s(i,2)*h(i,1);
    sxh[1] = s(i,2)*h(i,0) - s(i,0)*h(i,2);
    sxh[2] = s(i,0)*h(i,1) - s(i,1)*h(i,0);

    for(j=0; j<3; ++j) {
      snew(i,j) = s(i,j) + 0.5*dt*sxh[j];
    }

    for(j=0; j<3; ++j) {
      s(i,j) = s(i,j) + dt*sxh[j];
    }

    norm = 1.0/sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));

    for(j=0; j<3; ++j) {
      s(i,j) = s(i,j)*norm;
    }

    for(j=0; j<3; ++j) {
      rhs[j] = u(i,j) - omega_corr*(w(i,j) + alpha(i)*sxh[j]);
    }

    for(j=0; j<3; ++j) {
      wnew(i,j) = w(i,j) + 0.5*dt*rhs[j];
      w(i,j) = w(i,j) + dt*rhs[j];
    }
  }
  
  calculate_fields();

  for(i=0; i<nspins; ++i) {


    sxh[0] = s(i,1)*h(i,2) - s(i,2)*h(i,1);
    sxh[1] = s(i,2)*h(i,0) - s(i,0)*h(i,2);
    sxh[2] = s(i,0)*h(i,1) - s(i,1)*h(i,0);

    for(j=0; j<3; ++j) {
      s(i,j) = snew(i,j) + 0.5*dt*sxh[j];
    }
    
    norm = 1.0/sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));

    for(j=0; j<3; ++j) {
      s(i,j) = s(i,j)*norm;
    }
    for(j=0; j<3; ++j) {
      w(i,j) = wnew(i,j) + 0.5*dt*(u(i,j) - omega_corr*(w(i,j) + alpha(i)*sxh[j]));
    }
    //output.print("%e %e %e\n", w(i,0)/mus(i), w(i,1)/mus(i), w(i,2)/mus(i));
  }


  iteration++;
}
