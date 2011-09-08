#ifndef __DYNAMICSF_H__
#define __DYNAMICSF_H__

#include "physics.h"

#include <fftw3.h>

class DynamicSFPhysics : public Physics {
  public:
    DynamicSFPhysics()
    : initialised(false),
      qSpace(NULL),
      rSpaceFFT()
    {}
    
    ~DynamicSFPhysics();
    
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);

  private:
  bool              initialised;
  fftw_complex*     qSpace;
  fftw_plan         rSpaceFFT;


};

#endif /* __DYNAMICSF_H__ */
