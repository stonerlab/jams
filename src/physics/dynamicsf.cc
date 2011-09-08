#include "globals.h"
#include "dynamicsf.h"

void DynamicSFPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  initialised = true;
}

DynamicSFPhysics::~DynamicSFPhysics()
{

}

void  DynamicSFPhysics::run(double realtime, const double dt)
{
  using namespace globals;
  assert(initialised);
}

void DynamicSFPhysics::monitor(double realtime, const double dt)
{
  using namespace globals;
  assert(initialised);

}
