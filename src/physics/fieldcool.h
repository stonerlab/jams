#ifndef __FieldCool_H__
#define __FieldCool_H__

#include <fstream>
#include <libconfig.h++>
#include "array.h"

#include "physics.h"

class FieldCoolPhysics : public Physics {
  public:
    FieldCoolPhysics() 
      : initField(3,0.0),
        finalField(3,0.0),
        initTemp(0.0),
        finalTemp(0.0),
        coolTime(0.0),
        initialised(false)
    {}
    ~FieldCoolPhysics();
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);
  private:
    std::vector<double> initField;
    std::vector<double> finalField;
    double initTemp;
    double finalTemp;
    double coolTime;
    bool initialised;
};

#endif // __FieldCool_H__
