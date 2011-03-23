#ifndef __TTM_H__
#define __TTM_H__

#include <fstream>
#include <libconfig.h++>
#include "array.h"

#include "physics.h"

class TTMPhysics : public Physics {
  public:
    TTMPhysics() 
      : pumpTime(0.0),
        pumpStartTime(0.0),
        pumpTemp(0.0),
        pumpFluence(0.0),
        electronTemp(0.0),
        phononTemp(0.0),
        Ce(7.0E02),
        Cl(3.0E06),
        G(17.0E17),
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

    double pumpTime;
    double pumpStartTime;
    double pumpTemp;
    double pumpFluence;
    double electronTemp;
    double phononTemp;

    double Ce; // electron specific heat
    double Cl; // phonon specific heat
    double G;  // electron coupling constant

    std::ofstream TTMFile;

    bool initialised;
};

#endif // __TTM_H__
