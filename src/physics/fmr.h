#ifndef __FMR_H__
#define __FMR_H__

#include <fstream>
#include "array.h"

#include "physics.h"

class FMRPhysics : public Physics {
  public:
    FMRPhysics() 
      : initialised(false),
        ACFieldFrequency(0), 
        ACFieldStrength(3,0),
        DCFieldStrength(3,0),
        PSDFile(),
        PSDIntegral(0)
    {}
    ~FMRPhysics() {}
    void init();
    void run(double realtime);
    virtual void monitor(double realtime, const double dt);
  private:
    bool initialised;
    double ACFieldFrequency;
    std::vector<double> ACFieldStrength;
    std::vector<double> DCFieldStrength;
    std::ofstream PSDFile;
    Array<double> PSDIntegral;
};

#endif // __FMR_H__
