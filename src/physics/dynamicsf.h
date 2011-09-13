#ifndef __DYNAMICSF_H__
#define __DYNAMICSF_H__

#include "physics.h"

#include <vector>
#include <fftw3.h>

class DynamicSFPhysics : public Physics {
  public:
    DynamicSFPhysics()
    : initialised(false),
      timePointCounter(0),
      nTimePoints(0),
      rDim(3,0),
      upDim(3,0),
      upFac(3,1),
      qSpace(NULL),
      upSpace(NULL),
      tSpace(NULL),
      qSpaceFFT(),
      upSpaceFFT(),
      invUpSpaceFFT(),
      tSpaceFFT(),
      DSFFile(),
      freqIntervalSize(0)
    {}
    
    ~DynamicSFPhysics();
    
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);

  private:
  bool              initialised;
  int               timePointCounter;
  int               nTimePoints;
  std::vector<int>  rDim;
  std::vector<int>  upDim;
  std::vector<int>  upFac;
  fftw_complex*     qSpace;
  fftw_complex*     upSpace;
  fftw_complex*     tSpace;
  fftw_plan         qSpaceFFT;
  fftw_plan         upSpaceFFT;
  fftw_plan         invUpSpaceFFT;
  fftw_plan         tSpaceFFT;
  int               componentReal;
  int               componentImag;
  std::ofstream     DSFFile;
  double            freqIntervalSize;


};

#endif /* __DYNAMICSF_H__ */
