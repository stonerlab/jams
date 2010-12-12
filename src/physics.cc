#include "globals.h"
#include "physics.h"
#include "fmr.h"

void Physics::init() {
  if(initialised == true) {
    jams_error("Physics module is already initialised");
  }

  output.write("Initialising physics\n");

  initialised = true;
}

void Physics::run(double realtime) {
}

void Physics::monitor(const double realtime, const double dt) {

}

Physics* Physics::Create(PhysicsType type)
{
  switch(type){
    case FMR:
      return new FMRPhysics;
      break;
    default:
      jams_error("Unknown physics selected.");
  }
  return NULL;
}
