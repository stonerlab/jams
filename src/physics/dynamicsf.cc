#include "globals.h"
#include "dynamicsf.h"

#include <fftw3.h>
#include <string>
#include <map>

void DynamicSFPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  std::map<std::string,int> componentMap;
  componentMap["X"] = 0;
  componentMap["Y"] = 1;
  componentMap["Z"] = 2;

  std::string strImag, strReal;

  config.lookupValue("physics.componentReal",strReal);
  std::transform(strReal.begin(),strReal.end(),strReal.begin(),toupper);

  if(strReal != "X" || strReal != "Y" || strReal != "Z"){
    jams_error("Real Component for Fourier transform must be X,Y or Z");
  }
  componentReal = componentMap[strReal];

  if( config.exists("physics.componentImag") ) {
    config.lookupValue("physics.componentImag",strImag);
  
    if(strImag != "X" || strImag != "Y" || strImag != "Z"){
      jams_error("Imaginary Component for Fourier transform must be X,Y or Z");
    }
    std::transform(strImag.begin(),strImag.end(),strImag.begin(),toupper);
    
    componentImag = componentMap[strImag];
    output.write("  * Fourier transform component: (%s, i%s)\n",strReal.c_str(),strImag.c_str());
  } else {
    componentImag = -1; // dont use imaginary component
    output.write("  * Fourier transform component: %s\n",strReal.c_str());
  }


  // read lattice size
  if( config.exists("physics.SizeOverride") == true) {
    for(int i=0; i<3; ++i) {
      rDim[i] = phys["SizeOverride"][i];
    }
  output.write("  * Lattice size override [%d,%d,%d]\n",rDim[0],rDim[1],rDim[2]);
  }else{
    lattice.getDimensions(rDim[0],rDim[1],rDim[2]);
  }


  output.write("  * Allocating FFTW array...\n");
  qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*rDim[0]*rDim[1]*rDim[2]));

  output.write("  * Planning FFTW transform...\n");
  qSpaceFFT = fftw_plan_dft_3d(rDim[0],rDim[1],rDim[2],qSpace,qSpace,FFTW_FORWARD,FFTW_MEASURE);

  initialised = true;
}

DynamicSFPhysics::~DynamicSFPhysics()
{
  if(initialised == true){
    fftw_destroy_plan(qSpaceFFT);

    if(qSpace != NULL) {
      fftw_free(qSpace);
      qSpace = NULL;
    }
  }
}

void  DynamicSFPhysics::run(double realtime, const double dt)
{
  using namespace globals;
  assert(initialised);
}

void DynamicSFPhysics::monitor(double realtime, const double dt)
{
  using namespace globals;
  assert(initialised);


  // ASSUMPTION: Spin array is C-ordered in space
  if(componentImag == -1){
    for(int i=0; i<nspins; ++i){
      qSpace[i][0] = s(i,componentReal);
      qSpace[i][2] = 0.0;
    }
  } else {
    for(int i=0; i<nspins; ++i){
      qSpace[i][0] = s(i,componentReal);
      qSpace[i][2] = s(i,componentImag);
    }
  }

  fftw_execute(qSpaceFFT);

}
