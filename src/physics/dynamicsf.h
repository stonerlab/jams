#ifndef __DYNAMICSF_H__
#define __DYNAMICSF_H__

#include "physics.h"

#include <vector>
#include <fftw3.h>

class DynamicSFPhysics : public Physics {
  public:
    DynamicSFPhysics()
    : initialised(false),
      rDim(3,0),
      qSpace(NULL),
      rSpaceFFT()
    {}
    
    ~DynamicSFPhysics();
    
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);

  private:
  bool              initialised;
  std::vector<int>  rDim;
  fftw_complex*     qSpace;
  fftw_plan         rSpaceFFT;
  int               componentReal;
  int               componentImag;


};

#endif /* __DYNAMICSF_H__ */
