#include "globals.h"
#include "consts.h"

#include "heunllg.h"
#include "array2d.h"
#include <cmath>
#include "cblas.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


void HeunLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;
  
  // initialise base class
  Solver::initialise(argc,argv,idt);

  output.write("Initialising Heun LLG solver (CPU)\n");

  snew.resize(globals::nspins,3);

  sigma.resize(globals::nspins,3);

  temperature = 300.0;

  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*mus(i)) / (dt) );
    }
  }

  initialised = true;
}

void HeunLLGSolver::fields() {
  using namespace globals;
 
  // dscrmv below has beta=0.0 -> field array is zeroed
  // exchange
  const char transa[1] = {'N'};
  const char matdescra[6] = {'S','U','N','C','N','N'};
  if(Jij.nonzero() > 0) {
    jams_dcsrmv(transa,3*nspins,3*nspins,1.0,matdescra,Jij.ptrVal(),
        Jij.ptrCol(), Jij.ptrB(),Jij.ptrE(),s.ptr(),0.0,h.ptr()); 
  }

  // normalize by the gyroscopic factor
  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3;++j) {
      h(i,j) = (h(i,j)+w(i,j))*gyro(i);
    }
  }

  // multiply noise array by prefactor and add to fields
  //cblas_dsbmv(CblasColMajor,CblasUpper,3*nspins,0,sqrt(temperature),sigma.ptr(),1,w.ptr(),1,1.0,h.ptr(),1);

  //cblas_dtbmv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,nspins,0,gyro.ptr(),1,h.ptr(),1);
  //cblas_dtbmv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,nspins,0,gyro.ptr(),1,h.ptr()+nspins,1);
  //cblas_dtbmv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,nspins,0,gyro.ptr(),1,h.ptr()+2*nspins,1);
}

void HeunLLGSolver::run()
{
  using namespace globals;

  int i,j;
  double sxh[3], rhs[3];
  double norm;
  const double stmp = sqrt(temperature);

  // calculate noise
  for(i=0; i<nspins; ++i) {
    for(j=0; j<3; ++j) {
      w(i,j) = (rng.normal())*sigma(i,j)*stmp;
    }
  }
 
  fields();
  
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
  
  fields();

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
