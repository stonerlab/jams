#include <cmath>

#include "globals.h"

#include "fmr.h"

void FMRPhysics::init()
{
  using namespace globals;
  const libconfig::Setting& phys = config.lookup("physics");

  ACFieldFrequency = phys["ACFieldFrequency"];
  ACFieldFrequency = 2.0*M_PI*ACFieldFrequency;

  for(int i=0; i<3; ++i) {
    ACFieldStrength[i] = phys["ACFieldStrength"][i];
  }
  
  for(int i=0; i<3; ++i) {
    DCFieldStrength[i] = phys["DCFieldStrength"][i];
  }

  std::string fileName = "_psd.dat";
  fileName = seedname+fileName;
  PSDFile.open(fileName.c_str());

  PSDIntegral.resize(nspins);

  for(int i=0; i<nspins; ++i) {
    PSDIntegral(i) = 0;
  }

  initialised = true;

}

void FMRPhysics::run(const double realtime)
{
  
  for(int i=0; i<3; ++i) {
    globals::h_app[i] = DCFieldStrength[i] 
      + ACFieldStrength[i] * cos( ACFieldFrequency * realtime );
  }
  
}

void FMRPhysics::monitor(const double realtime, const double dt)
{
  using namespace globals;

  double pAverage = 0.0;

  for(int i=0; i<nspins; ++i) {
    const double sProjection = 
      s(i,0)*ACFieldStrength[0] + s(i,1)*ACFieldStrength[1] + s(i,2)*ACFieldStrength[2];

    PSDIntegral(i) += sProjection * sin( ACFieldFrequency * realtime ) * dt;

    pAverage += fabs(PSDIntegral(i) * ( ACFieldFrequency * mus(i) ) / realtime);
  }

  PSDFile << realtime << "\t" << pAverage/nspins << "\n";

}