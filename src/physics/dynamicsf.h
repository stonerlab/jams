#ifndef __DYNAMICSF_H__
#define __DYNAMICSF_H__

#include "physics.h"

class DynamicSFPhysics : public Physics {
  public:
    DynamicSFPhysics()
    : initialised(false)
    {}
    
    ~DynamicSFPhysics();
    
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);

  private:
  bool initialised;

};

#endif /* __DYNAMICSF_H__ */
