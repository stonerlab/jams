#include <cmath>
#include <iomanip>
#include <libconfig.h++>

#include "globals.h"

#include "fieldcool.h"

void FieldCoolPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * Field cooled physics module\n");  

  initTemp = phys["InitialTemperature"];
  finalTemp = phys["FinalTemperature"];
  
  for(int i=0; i<3; ++i){
    initField[i] = phys["InitialField"][i];
    finalField[i] = phys["FinalField"][i];
  }

  coolTime = phys["CoolTime"];
    
  for(int i=0; i<3; ++i) {
    globals::h_app[i] += initField[i];
  }

  globalTemperature = initTemp;

  initialised = true;

}

FieldCoolPhysics::~FieldCoolPhysics()
{
}

void FieldCoolPhysics::run(const double realtime, const double dt)
{
  using namespace globals;

  double fieldRate[3];
  for(int i=0; i<3; ++i){
    fieldRate[i] = ((finalField[i]-initField[i])*dt)/coolTime;
  }
  const double tempRate = ((finalTemp-initTemp)*dt)/coolTime;

  if( realtime < coolTime ) {

    for(int i=0; i<3; ++i) {
      globals::h_app[i] += fieldRate[i];
    }

    globalTemperature += tempRate;
  }

  
}

void FieldCoolPhysics::monitor(const double realtime, const double dt)
{}
