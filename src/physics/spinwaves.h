#ifndef __SPINWAVES_H__
#define __SPINWAVES_H__

#include <fstream>
#include <vector>
#include <libconfig.h++>
#include <fftw3.h>
#include "physics.h"

class SpinwavesPhysics : public Physics {
  public:
    SpinwavesPhysics()
      : dim(3,0),
        FFTPlan(),
        FFTArray(NULL),
        SPWFile(),
        SPDFile(),
        typeOverride(),
        initialised(false),
        spinDump(false),        
        PulseDuration(0),
        PulseTemperature(0),
        PulseCount(0),
        PulseTotal(0),
        FieldStrength(3,0)
      {}

    ~SpinwavesPhysics();

    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);

  private:
    std::vector<int> dim;
    fftw_plan       FFTPlan;
    fftw_complex*   FFTArray;
    std::ofstream   SPWFile;
    std::ofstream   SPDFile;
    std::vector<int> typeOverride;
    bool initialised;
    bool spinDump;
    double PulseDuration;
    double PulseTemperature;
    int    PulseCount;
    int    PulseTotal;
    std::vector<double> FieldStrength;

};
#endif /* __SPINWAVES_H__ */
