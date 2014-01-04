#ifndef JAMS_PHYSICS_SQUARE_H
#define JAMS_PHYSICS_SQUARE_H

#include <fstream>
#include <libconfig.h++>

#include "physics.h"

class SquarePhysics : public Physics {
  public:
    SquarePhysics() 
      : initialised(false),
        PulseDuration(0),
        PulseCount(0),
        PulseTotal(0),
        FieldStrength(3,0)
    {}
    ~SquarePhysics();
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);
  private:
    bool initialised;
    double PulseDuration;
    int    PulseCount;
    int    PulseTotal;
    std::vector<double> FieldStrength;
};

#endif // JAMS_PHYSICS_SQUARE_H
