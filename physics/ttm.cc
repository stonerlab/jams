#include <cmath>
#include <iomanip>
#include <libconfig.h++>

#include <math/functions.h>

#include "globals.h"

#include "ttm.h"

void TTMPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * Two temperature model physics module\n");  

  phononTemp = phys["InitialTemperature"];
  electronTemp = phononTemp;
  
  sinkTemp = phononTemp;

  const libconfig::Setting& laserPulseConfig = ::config.lookup("physics.laserPulses");

  const int nLaserPulses = laserPulseConfig.getLength();

  pulseWidth.resize(nLaserPulses);
  pulseFluence.resize(nLaserPulses);
  pulseStartTime.resize(nLaserPulses);

  for (int i=0; i!=nLaserPulses; ++i) {
    pulseWidth(i) = laserPulseConfig[i]["width"];
    pulseFluence(i) = laserPulseConfig[i]["fluence"];
    pulseFluence(i) = pumpPower(pulseFluence(i));
    pulseStartTime(i) = laserPulseConfig[i]["t_start"];
  }

  // if these settings don't exist, the defaults will be left in
  phys.lookupValue("Ce",Ce);
  phys.lookupValue("Cl",Cl);
  phys.lookupValue("Gep",G);
  phys.lookupValue("Gps",Gsink);

  for(int i=0; i<3; ++i) {
    reversingField[i] = phys["ReversingField"][i];
  }


  std::string fileName = "_ttm.dat";
  fileName = seedname+fileName;
  TTMFile.open(fileName.c_str());

  TTMFile << std::setprecision(8);
  TTMFile << "# t [s]\tT_el [K]\tT_ph [K]\tLaser [arb/]\n";

  initialised = true;

}

TTMPhysics::~TTMPhysics()
{
  TTMFile.close();
}

void TTMPhysics::run(const double realtime, const double dt)
{
  using namespace globals;
  using namespace jblib;


  for(int i=0; i<3; ++i) {
    globals::h_app[i] = reversingField[i];
  }


  pumpTemp = 0.0;
  for (int i=0,iend=pulseFluence.size(); i!=iend; ++i) {
    const double relativeTime = (realtime-pulseStartTime(i));
    if( (relativeTime > 0.0) && (relativeTime <= 10*pulseWidth(i)) ) {
      pumpTemp = pumpTemp + pulseFluence(i)*exp(-square((relativeTime-3*pulseWidth(i))/(pulseWidth(i))));
    }
  }

  electronTemp = electronTemp + ((-G*(electronTemp-phononTemp)+pumpTemp)*dt)/(Ce*electronTemp);
  phononTemp   = phononTemp   + (( G*(electronTemp-phononTemp)-Gsink*(phononTemp-sinkTemp))*dt)/(Cl);

  globalTemperature = electronTemp;


}

void TTMPhysics::monitor(const double realtime, const double dt)
{
  using namespace globals;

  TTMFile << realtime << "\t" << electronTemp << "\t" << phononTemp << "\t" << pumpTemp << "\n";

}
