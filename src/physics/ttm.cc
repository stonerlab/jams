#include <cmath>
#include <iomanip>
#include <libconfig.h++>

#include "globals.h"

#include "ttm.h"

void TTMPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  phononTemp = phys["InitialTemperature"];
  electronTemp = phononTemp;

  // unitless according to Tom's code!
  pumpFluence = phys["PumpFluence"];
  pumpFluence = pumpPower(pumpFluence);

  // width of gaussian heat pulse in seconds
  pumpTime = phys["PumpTime"];

  pumpStartTime = phys["PumpStartTime"];

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

  const double relativeTime = (realtime-pumpStartTime);


  if( relativeTime > 0.0 ) {

    for(int i=0; i<3; ++i) {
      globals::h_app[i] = reversingField[i];
    }
    if( relativeTime <= 10*pumpTime ) {
      pumpTemp = pumpFluence*exp(-((relativeTime-3*pumpTime)/(pumpTime))*((relativeTime-3*pumpTime)/(pumpTime)));
    } else {
      pumpTemp = 0.0;
    }

    electronTemp = electronTemp + ((-G*(electronTemp-phononTemp)+pumpTemp)*dt)/(Ce*electronTemp);
    phononTemp   = phononTemp   + (( G*(electronTemp-phononTemp)         )*dt)/(Cl);
  }

  
}

void TTMPhysics::monitor(const double realtime, const double dt)
{
  using namespace globals;

  TTMFile << realtime << "\t" << electronTemp << "\t" << phononTemp << "\t" << pumpTemp << "\n";

}
