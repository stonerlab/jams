#include <cmath>
#include <libconfig.h++>

#include "globals.h"

#include "square.h"

void SquarePhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * Square physics module\n");

  PulseDuration = phys["PulseDuration"];
  PulseTotal    = phys["PulseTotal"];

  for(int i=0; i<3; ++i) {
    FieldStrength[i] = phys["FieldStrength"][i];
  }

  PulseCount = 1;
  
  initialised = true;

}

SquarePhysics::~SquarePhysics()
{
}

void SquarePhysics::run(const double realtime, const double dt)
{
  using namespace globals;

  if(realtime > (PulseDuration*PulseCount)){
    PulseCount++;
  }
  
  if(realtime < (PulseDuration*(PulseTotal))){
    if(PulseCount%2 == 0) {
      for(int i=0; i<3; ++i) {
        globals::h_app[i] = -FieldStrength[i];
      }
    }else{
      for(int i=0; i<3; ++i) {
        globals::h_app[i] = FieldStrength[i];
      }
    }
  }else{
    for(int i=0; i<3; ++i) {
      globals::h_app[i] = 0.0; 
    }
  }
  
}

void SquarePhysics::monitor(const double realtime, const double dt)
{
}
