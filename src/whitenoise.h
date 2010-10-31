#ifndef __WHITENOISE_H__
#define __WHITENOISE_H__

#include "noise.h"
#include "array2d.h"

class WhiteNoise : public Noise {
  public:
    WhiteNoise() : half(false), sigma(0,0) {}
    ~WhiteNoise() {}
    void initialise(double dt);
    void run();
  private:
    bool half;
    Array2D<double> sigma;
};

#endif // __WHITENOISE_H__
