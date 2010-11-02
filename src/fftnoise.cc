#include "fftnoise.h"
#include "globals.h"
#include "fields.h"
#include "consts.h"
#include <cmath>

double FFTNoise::kernel_exp(const double omega, const double idt, const int mu, const int n) {
  assert(n%2 == 0);

  // return 1.0; // <- white noise

  // only for ohmic nosie (otherwise delta power is needed)
  if(mu <= n/2 || mu == n) {
    return sqrt(2.0*n*idt*gamma_d*pow(((2*pi*mu)/(n*idt*omega)),(delta-1))*exp( -(2.0*pi*mu)/(n*idt*omega_c) ));
  } else {
    return kernel_exp(omega,idt,n-mu,n);
  }
}

void FFTNoise::initialise(int argc, char **argv, double idt){
  using namespace globals;

  Solver::initialise(argc,argv,idt);

  output.write("Initialising FFT Noise\n");

  temperature=0.01;
  
  delta = 1.0;
  gamma_d = alpha(0);
  omega_t = 1.0/dt; //568.180;
  omega_c = omega_t/100.0;

  const int nsteps = 1048576/16; //(2^20)

  // resize arrays
  mem.resize(nspins,3);
  memnew.resize(nspins,3);
  sigma.resize(nspins,3);
  snew.resize(nspins,3);
  s0.resize(nspins,3);
  
  phase = static_cast<fftw_complex*>(fftw_malloc(nsteps*3*sizeof(fftw_complex)));
  spec.resize(nsteps,3);
  corr.resize(nsteps,3);
  
  // create plans
  fftw_plan corr_plan = fftw_plan_r2r_2d(nsteps/2,3,corr.ptr(),corr.ptr(),FFTW_REDFT01,FFTW_REDFT01,FFTW_ESTIMATE);
  fftw_plan spec_plan = fftw_plan_dft_c2r_2d(nsteps,3,phase,spec.ptr(),FFTW_ESTIMATE);

  // initialise arrays
  for(int i=0;i<nspins;++i) {
    for(int j=0;j<3;++j) {
      mem(i,j) = 0.0;
      memnew(i,j) = 0.0;
      s0(i,j) = s(i,j);
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*mus(i)) / (dt) );
    }
  }

  // create xi(omega) spectrum from the memory kernel
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

  // create power spectrum S(omega) (must be done before fftw_execute
  // calls)
  const double d2pi = 1.0/(2.0*pi*nsteps);
  for(int i=0; i<nsteps; ++i) {
    for(int j=0; j<3; ++j) {
      corr(i,j) = (phase[i*3+j][0]*phase[i*3+j][0] + phase[i*3+j][1]*phase[i*3+j][1]);
      corr(i,j) = d2pi*mus(0)*corr(i,j)*corr(i,j);
    }
  }
  
  // output random phase spectrum
  std::ofstream outfile("spectrum.dat");
  outfile << "#  i | omega | Re xi_x | Im xi_x | Re xi_y | Im xi_y | Re xi_z | Im xi_z "<<std::endl;
  for(int i=0; i<nsteps/2; ++i) {
    outfile << i << "\t" << (2*pi*i*gamma_electron_si)/(nsteps*dt) << "\t";
    for(int j=0;j<3;++j) {
      outfile << phase[i*3+j][0] << "\t" << phase[i*3+j][1] << "\t";
    }
    outfile << std::endl;
  }
  for(int i=nsteps/2; i<nsteps; ++i) {
    outfile << i << "\t" << -(2*pi*(nsteps-i)*gamma_electron_si)/(nsteps*dt) << "\t";
    for(int j=0;j<3;++j) {
      outfile << phase[i*3+j][0] << "\t" << phase[i*3+j][1] << "\t";
    }
    outfile << std::endl;
  }
  outfile.close();


  // FT to xi(omega) -> xi(t)
  fftw_execute(spec_plan);
  
  // normalise FT
  const double dnsteps = 1.0/sqrt(nsteps);
  for(int i=0; i<nsteps; ++i) {
    for(int j=0;j<3;++j) {
      spec(i,j) = spec(i,j)*dnsteps;
    }
  }
  
  // FT S(omega) -> C(tau)
  // (power spectrum to autocorrelation function - Wiener-Khinchin
  // theorem)
  fftw_execute(corr_plan);
  // does not need normlising because S(omega) was normalised in
  // it's definition

  // output random numbers and correlation function x(t), <xi(0)xi(t)>
  outfile.open("correlation.dat");
  outfile << "#  i | t | xi_x | xi_y | xi_z | <xi(0)xi(t)>_x | <xi(0)xi(t)>_y | <xi(0)xi(t)>_z "<<std::endl;
  for(int i=0; i<nsteps; ++i) {
    outfile << i << "\t";
    for(int j=0;j<3;++j) {
      outfile << spec(i,j) << "\t";
    }
    for(int j=0;j<3;++j) {
      outfile << corr(i,j) << "\t";
    }
    outfile << std::endl;
  }
  outfile.close();



  fftw_free(phase);

}

void FFTNoise::run() {
  using namespace globals;
  
  int i,j;
  double sxh[3], rhs[3];
  double norm;
 
  for(i=0; i<nspins; ++i) {
  
    for(j=0; j<3; ++j) {
      w(i,j) = sqrt(temperature)*sigma(i,j)*spec(iteration,j) 
              - alpha(i) * ( s(i,j)*corr(0,j) -corr(iteration,j)*s0(i,j) ) + alpha(i)*mem(i,j);
      rhs[j] = s(i,j)*(corr(iteration+1,j) - corr(iteration,j));
      memnew(i,j) = mem(i,j) + 0.5*rhs[j];
      mem(i,j) = mem(i,j) + rhs[j];
    }
  }

  calculate_fields();
  
//  output.write("%e %e %e\n",rhs[0],rhs[1],rhs[2]);
  
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
  
  for(i=0; i<nspins; ++i) {
    
    for(j=0; j<3; ++j) {
      w(i,j) = sqrt(temperature)*sigma(i,j)*spec(iteration,j) 
              - alpha(i) * ( s(i,j)*corr(0,j) -corr(iteration,j)*s0(i,j) ) + alpha(i)*mem(i,j);
      rhs[j] = s(i,j)*(corr(iteration+1,j) - corr(iteration,j));
      mem(i,j) = memnew(i,j) + 0.5*rhs[j];
    }
  }
  
  calculate_fields();
  //output.write("%e %e %e\n",h(0,0),h(0,1),h(0,2));

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
