#ifndef JAMS_PHYSICS_FMR_H
#define JAMS_PHYSICS_FMR_H

#include <fstream>
#include <libconfig.h++>

#include "physics.h"

#include <containers/array.h>

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
    ~FMRPhysics();
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);
  private:
    bool initialised;
    double ACFieldFrequency;
    std::vector<double> ACFieldStrength;
    std::vector<double> DCFieldStrength;
    std::ofstream PSDFile;
    jblib::Array<double,1> PSDIntegral;
};

#endif // JAMS_PHYSICS_FMR_H
