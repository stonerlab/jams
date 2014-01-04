#ifndef JAMS_PHYSICS_TTM_H
#define JAMS_PHYSICS_TTM_H

#include <fstream>
#include <libconfig.h++>
#include <containers/array.h>

#include "physics.h"

class TTMPhysics : public Physics {
  public:
    TTMPhysics() 
      : pulseWidth(0),
        pulseFluence(0),
        pulseStartTime(0),
        pumpTemp(0.0),
        electronTemp(0.0),
        phononTemp(0.0),
        sinkTemp(0.0),
        reversingField(3,0.0),
        Ce(7.0E02),
        Cl(3.0E06),
        G(17.0E17),
        Gsink(17.0E14),
        TTMFile(),
        initialised(false)
    {}
    ~TTMPhysics();
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);
  private:

    // calculation of pump power which is linear with input approx
    // electron temperature
    double pumpPower(double &pF){return (1.152E20*pF);}

    jblib::Array<double,1> pulseWidth;
    jblib::Array<double,1> pulseFluence;
    jblib::Array<double,1> pulseStartTime;
    double pumpTemp;
    double electronTemp;
    double phononTemp;
    double sinkTemp;
    std::vector<double> reversingField;

    double Ce; // electron specific heat
    double Cl; // phonon specific heat
    double G;  // electron coupling constant
    double Gsink;

    std::ofstream TTMFile;

    bool initialised;
};

#endif // JAMS_PHYSICS_TTM_H
