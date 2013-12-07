#ifndef __TTM_H__
#define __TTM_H__

#include <fstream>
#include <libconfig.h++>
#include "../../../jbLib/containers/Array.h"

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

    jbLib::Array<double,1> pulseWidth;
    jbLib::Array<double,1> pulseFluence;
    jbLib::Array<double,1> pulseStartTime;
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

#endif // __TTM_H__
