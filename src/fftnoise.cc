#include "fftnoise.h"
#include "globals.h"
#include "consts.h"
#include <cmath>

double FFTNoise::kernel_exp(const double omega, const double dt, const int mu, const int n) {
  assert(n%2 == 0);

  // return 1.0; // <- white noise

  // only for ohmic nosie (otherwise delta power is needed)
  if(mu <= n/2 || mu == n) {
    return sqrt(2.0*n*dt*gamma_d*pow(((2*pi*mu)/(n*dt*omega)),(delta-1))*exp( -(2.0*M_PI*mu)/(n*dt*omega_c) ));
  } else {
    return kernel_exp(omega,dt,n-mu,n);
  }
}

void FFTNoise::initialise(double dt) {
  using namespace globals;

  output.write("Initialising FFT Noise\n");

  int nsteps = 1048576; //(2^20)

  sigma.resize(nspins,3);

  delta = 1.0;
  gamma_d = 1.0;
  omega_t = 1.0/dt;
  omega_c = 1.0/(100*dt);

  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*mus(i)) / (dt) );
    }
  }

  phase = (fftw_complex*)fftw_malloc(nsteps*3*sizeof(fftw_complex));

  spec.resize(nsteps,3);

  spec_plan = fftw_plan_dft_c2r_2d(nspins,3,phase,spec.ptr(),FFTW_ESTIMATE);

  for(int j=0;j<3;++j) {
    phase[j][0] = kernel_exp(omega_t,dt,nsteps,nsteps);
    phase[j][1] = kernel_exp(omega_t,dt,nsteps,nsteps);
  }

  for(int i=1;i<nsteps;++i) {
    for(int j=0;j<3;++j) {
      phase[i*3+j][0] = sqrt(kernel_exp(omega_t,dt,i,nsteps))*sqrt(0.5)*rng.normal();
      phase[i*3+j][1] = sqrt(kernel_exp(omega_t,dt,i,nsteps))*sqrt(0.5)*rng.normal();
    }
  }

  std::ofstream outfile("spectrum.dat");

  outfile << "#  i | Re(Px) | Im(Px) | Re(Py) | Im(Py) | Re(Pz) | Im(Pz) "<<std::endl;

  for(int i=0; i<nsteps; ++i) {
    outfile << i << "\t";
    for(int j=0;j<3;++j) {
      outfile << phase[i*3+j][0] << "\t" << phase[i*3+j][1] << "\t";
    }
    outfile << std::endl;
  }

  outfile.close();

  fftw_execute(spec_plan);

  fftw_free(phase);

  // FFTW is an unnormalized transform -> normalise by sqrt(nsteps)
  for(int i=0; i<nsteps; ++i) {
    for(int j=0;j<3;++j) {
      spec(i,j) = spec(i,j)/sqrt(nsteps);
    }
  }

  outfile.open("correlation.dat");
  
  outfile << "#  i | Cx | Cy | Cz | rng.normal | kernel "<<std::endl;

  double corr_sum = 0.0;


  int n0 = nsteps/4;
  // can do 3/4 of the numbers
  for(int i=0; i<(3*n0); ++i) {

    double kernel = 0.0;
    for(int j=0; j<n0; ++j){
      kernel += spec(j+i,0)*spec(j,0);
    }
    kernel /= static_cast<double>(n0+1);

    outfile << i << "\t" << spec(i,0) << "\t" << spec(i,1) << "\t" << spec(i,2) << "\t";
    outfile << rng.normal() << "\t";
    outfile << kernel << "\n";

  }
  outfile.close();


}

void FFTNoise::run() {

}
