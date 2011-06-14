#include "globals.h"
#include "physics.h"
#include "fmr.h"
#include "ttm.h"
#include "spinwaves.h"
#include "empty.h"

#include <libconfig.h++>

void Physics::init(libconfig::Setting &phys) {
  if(initialised == true) {
    jams_error("Physics module is already initialised");
  }

  initialised = true;
}

void Physics::run(double realtime, const double dt) {
}

void Physics::monitor(const double realtime, const double dt) {

}

Physics* Physics::Create(PhysicsType type)
{
  switch(type){
    case FMR:
      return new FMRPhysics;
      break;
    case TTM:
      return new TTMPhysics;
      break;
    case SPINWAVES:
      return new SpinwavesPhysics;
      break;
    case EMPTY:
      return new EmptyPhysics;
      break;
    default:
      jams_error("Unknown physics selected.");
  }
  return NULL;
}
