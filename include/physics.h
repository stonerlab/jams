#ifndef __PHYSICS_H__
#define __PHYSICS_H__

enum PhysicsType{ FMR };

class Physics
{
  public:
    Physics()
      : initialised(false)
    {}

    virtual ~Physics(){}

    virtual void init();
    virtual void run(const double realtime);
    virtual void monitor(const double realtime, const double dt);

    static Physics* Create(PhysicsType type);

  protected:
    bool initialised;

};

#endif // __PHYSICS_H__