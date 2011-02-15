#ifndef __EMPTY_H__
#define __EMPTY_H__

#include "physics.h"

class EmptyPhysics : public Physics {
  public:
    EmptyPhysics() 
      : initialised(true)
    {}
    ~EmptyPhysics();
    void init();
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);
  private:
    bool initialised;
};

#endif // __EMPTY_H__
