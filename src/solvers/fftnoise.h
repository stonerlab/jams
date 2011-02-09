#ifndef __FFTNOISE_H__
#define __FFTNOISE_H__

#include "solver.h"
#include "array2d.h"
#include <fftw3.h>

class FFTNoise : public Solver {
  public:
    FFTNoise() 
      : delta(1.0),
        gamma_d(0), 
        omega_t(0),
        omega_c(0),
        snew(0,0),
        s0(0,0),
        sigma(0,0),
        mem(0,0),
        memnew(0,0),
        phase(NULL),
        spec(0,0),
        corr(0,0)
    {}
    ~FFTNoise() {}
    void initialise(int argc, char **argv, double idt);
    void run();
    void syncOutput();

  private:
    double delta;
    double gamma_d;
    double omega_t;
    double omega_c;

    Array2D<double> snew;
    Array2D<double> s0;
    Array2D<double> sigma;
    Array2D<double> mem;
    Array2D<double> memnew;
    
    fftw_complex* phase;
    Array2D<double> spec;
    Array2D<double> corr;

    double kernel_exp(const double omega, const double idt, const int mu, const int n);
};

#endif // __FFTNOISE_H__
