#include "physics/mfpt.h"

#include <libconfig.h++>

#include <cmath>
#include <string>

#include "core/globals.h"

void MFPTPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * MFPT physics module\n");

  std::string fileName = "_mfpt.dat";
  fileName = seedname+fileName;
  MFPTFile.open(fileName.c_str());

  maskArray.resize(nspins);

  for(int i=0; i<nspins; ++i){
      maskArray[i] = true;
  }

  initialised = true;

}

MFPTPhysics::~MFPTPhysics()
{
  MFPTFile.close();
}

void MFPTPhysics::run(const double realtime, const double dt)
{
  using namespace globals;
  
  
}

void MFPTPhysics::monitor(const double realtime, const double dt)
{
  using namespace globals;
  
  for(int i=0; i<nspins; ++i){
    if(s(i,2) < 0.0 && maskArray[i] == true){
        MFPTFile << realtime << "\n";
        maskArray[i] = false;
    }
  }
}
