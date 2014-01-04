#include "solvers/heunllg.h"

#include <cmath>

#include "core/consts.h"
#include "core/fields.h"
#include "core/globals.h"

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
  J1ij_s.convertMAP2CSR();
  J1ij_t.convertMAP2CSR();
  J2ij_s.convertMAP2CSR();
  J2ij_t.convertMAP2CSR();

  output.write("  * J1ij Scalar matrix memory (CSR): %f MB\n",J1ij_s.calculateMemory());
  output.write("  * J1ij Tensor matrix memory (CSR): %f MB\n",J1ij_t.calculateMemory());
  output.write("  * J2ij Scalar matrix memory (CSR): %f MB\n",J2ij_s.calculateMemory());
  output.write("  * J2ij Tensor matrix memory (CSR): %f MB\n",J2ij_t.calculateMemory());

  snew.resize(nspins,3);
  sigma.resize(nspins,3);
  eng.resize(nspins,3);
  
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

void HeunLLGSolver::calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s)
{
  using namespace globals;

  e1_s = 0.0; e1_t = 0.0; e2_s = 0.0; e2_t = 0.0;

  if(J1ij_s.nonZero() > 0) {
    std::fill(eng.data(),eng.data()+nspins3,0.0); 
    calc_scalar_bilinear(J1ij_s.valPtr(),J1ij_s.colPtr(),J1ij_s.ptrB(),J1ij_s.ptrE(),eng);
    for(int i=0; i<nspins; ++i){
      e1_s = e1_s + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    e1_s = e1_s/nspins;
  } 
  if(J1ij_t.nonZero() > 0) {
    std::fill(eng.data(),eng.data()+nspins3,0.0); 
    calc_tensor_bilinear(J1ij_t.valPtr(),J1ij_t.colPtr(),J1ij_t.ptrB(),J1ij_t.ptrE(),eng);
    for(int i=0; i<nspins; ++i){
      e1_t = e1_t + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    e1_t = e1_t/nspins;
  }
  if(J2ij_s.nonZero() > 0) {
    std::fill(eng.data(),eng.data()+nspins3,0.0); 
    calc_scalar_biquadratic(J2ij_s.valPtr(),J2ij_s.colPtr(),J2ij_s.ptrB(),J2ij_s.ptrE(),eng);
    for(int i=0; i<nspins; ++i){
      e2_s = e2_s + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    
    e2_s = 0.5*e2_s/nspins;
  }
  if(J2ij_t.nonZero() > 0) {
    std::fill(eng.data(),eng.data()+nspins3,0.0); 
    calc_tensor_biquadratic(J2ij_t.valPtr(),J2ij_t.colPtr(),J2ij_t.ptrB(),J2ij_t.ptrE(),eng);
    for(int i=0; i<nspins; ++i){
      e2_t = e2_t + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    
    e2_t = 0.5*e2_t/nspins;
  }

}
