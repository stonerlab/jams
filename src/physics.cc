#include "globals.h"
#include "physics.h"
#include "fmr.h"
#include "empty.h"

#include <libconfig.h++>

void Physics::init(libconfig::Setting &phys) {
  if(initialised == true) {
    jams_error("Physics module is already initialised");
  }

  output.write("Initialising physics\n");

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
    case EMPTY:
      return new EmptyPhysics;
      break;
    default:
      jams_error("Unknown physics selected.");
  }
  return NULL;
}
