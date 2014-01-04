#include "physics/fmr.h"

#include <libconfig.h++>

#include <cmath>
#include <string>

#include "core/globals.h"


void FMRPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * FMR physics module\n");

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

FMRPhysics::~FMRPhysics()
{
  PSDFile.close();
}

void FMRPhysics::run(const double realtime, const double dt)
{
  using namespace globals;
  
  const double cosValue = cos( ACFieldFrequency * realtime );
  const double sinValue = sin( ACFieldFrequency * realtime );

  for(int i=0; i<3; ++i) {
    globals::h_app[i] = DCFieldStrength[i] 
      + ACFieldStrength[i] * cosValue;
  }
  
  for(int i=0; i<nspins; ++i) {
    const double sProjection = 
      s(i,0)*ACFieldStrength[0] + s(i,1)*ACFieldStrength[1] + s(i,2)*ACFieldStrength[2];

    PSDIntegral(i) += sProjection * sinValue * dt;
  }
  
}

void FMRPhysics::monitor(const double realtime, const double dt)
{
  using namespace globals;

  double pAverage = 0.0;

  for(int i=0; i<nspins; ++i) {
    pAverage += fabs(PSDIntegral(i) * ( ACFieldFrequency * mus(i) ) / realtime);
  }

  PSDFile << realtime << "\t" << pAverage/nspins << "\n";

}
