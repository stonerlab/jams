#ifndef __FFTNOISE_H__
#define __FFTNOISE_H__

#include "noise.h"
#include "array2d.h"
#include <fftw3.h>

class FFTNoise : public Noise {
  public:
    FFTNoise() 
      : delta(1.0),
        gamma_d(0), 
        omega_t(0),
        omega_c(0),
        sigma(0,0),
        phase(NULL),
        spec(0,0),
        spec_plan(NULL)
    {}
    ~FFTNoise() {}
    void initialise(double dt);
    void run();
  private:
    double delta;
    double gamma_d;
    double omega_t;
    double omega_c;

    Array2D<double> sigma;
    
    fftw_complex* phase;
    Array2D<double> spec;
    fftw_plan spec_plan;

    double kernel_exp(const double omega, const double dt, const int mu, const int n);
};

#endif // __FFTNOISE_H__
