#ifndef __PHYSICS_H__
#define __PHYSICS_H__

#include <libconfig.h++>

enum PhysicsType{ EMPTY, FMR, TTM, SPINWAVES, SQUARE, DYNAMICSF};

class Physics
{
  public:
    Physics()
      : initialised(false)
    {}

    virtual ~Physics(){}

    virtual void init(libconfig::Setting &phys);
    virtual void run(const double realtime, const double dt);
    virtual void monitor(const double realtime, const double dt);

    static Physics* Create(PhysicsType type);

  protected:
    bool initialised;

};

#endif // __PHYSICS_H__
