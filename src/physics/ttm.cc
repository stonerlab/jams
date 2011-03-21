#include <cmath>
#include <iomanip>
#include <libconfig.h++>

#include "globals.h"

#include "ttm.h"

void TTMPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  electronTemp = phys["InitialTemperature"];

  // unitless according to Tom's code!
  pumpFluence = phys["PumpFluence"];
  pumpFluence = pumpPower(pumpFluence);

  // width of gaussian heat pulse in seconds
  pumpTime = phys["PumpTime"];


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

  if(realtime < pumpTime) {
    pumpTemp = pumpFluence*exp(-((realtime-3*pumpTime)/(pumpTime))*((realtime-3*pumpTime)/(pumpTime)));
  } else {
    pumpTemp = 0.0;
  }

  electronTemp = electronTemp + ((-G*(electronTemp-phononTemp)+pumpTemp)*dt)/(Ce*electronTemp);
  phononTemp   = phononTemp   + (( G*(electronTemp-phononTemp)         )*dt)/(Cl);

  
}

void TTMPhysics::monitor(const double realtime, const double dt)
{
  using namespace globals;

  TTMFile << realtime << "\t" << electronTemp << "\t" << phononTemp << "\t" << pumpTemp << "\n";

}
