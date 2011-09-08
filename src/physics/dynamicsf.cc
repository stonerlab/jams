#include "globals.h"
#include "dynamicsf.h"

#include <fftw3.h>
#include <string>
#include <map>

void DynamicSFPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  const double sampletime = config.lookup("sim.t_out");
  const double runtime = config.lookup("sim.t_run");
  nTimePoints = runtime/sampletime;

  freqIntervalSize = (2.0*M_PI)/(sampletime);

  std::map<std::string,int> componentMap;
  componentMap["X"] = 0;
  componentMap["Y"] = 1;
  componentMap["Z"] = 2;

  std::string strImag, strReal;

  config.lookupValue("physics.componentReal",strReal);
  std::transform(strReal.begin(),strReal.end(),strReal.begin(),toupper);

//   if(strReal != "X" || strReal != "Y" || strReal != "Z"){
//     jams_error("Real Component for Fourier transform must be X,Y or Z");
//   }
  componentReal = componentMap[strReal];

  if( config.exists("physics.componentImag") ) {
    config.lookupValue("physics.componentImag",strImag);
  
//     if(strImag != "X" || strImag != "Y" || strImag != "Z"){
//       jams_error("Imaginary Component for Fourier transform must be X,Y or Z");
//     }
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


  output.write("  * Allocating FFTW arrays...\n");
  qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*rDim[0]*rDim[1]*rDim[2]));
  tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*((rDim[2]/2)+1)));

  output.write("  * Planning FFTW transform...\n");
  qSpaceFFT = fftw_plan_dft_3d(rDim[0],rDim[1],rDim[2],qSpace,qSpace,FFTW_FORWARD,FFTW_MEASURE);
  tSpaceFFT = fftw_plan_dft_2d(nTimePoints,((rDim[2]/2)+1),tSpace,tSpace,FFTW_FORWARD,FFTW_MEASURE);

  std::string filename = "_dsf.dat";
  filename = seedname+filename;
  DSFFile.open(filename.c_str());

  initialised = true;
}

DynamicSFPhysics::~DynamicSFPhysics()
{
  if(initialised == true){
    fftw_destroy_plan(qSpaceFFT);
    fftw_destroy_plan(tSpaceFFT);

    if(qSpace != NULL) {
      fftw_free(qSpace);
      qSpace = NULL;
    }
    if(tSpace != NULL) {
      fftw_free(tSpace);
      tSpace = NULL;
    }
  }

  DSFFile.close();
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
      qSpace[i][1] = 0.0;
    }
  } else {
    for(int i=0; i<nspins; ++i){
      qSpace[i][0] = s(i,componentReal);
      qSpace[i][1] = s(i,componentImag);
    }
  }

  fftw_execute(qSpaceFFT);

  // average over -q +q
  
  // cant average zero mode
  int tIdx = ((rDim[2]/2)+1)*timePointCounter;
  tSpace[tIdx][0] = (qSpace[0][0]*qSpace[0][0] + qSpace[0][1]*qSpace[0][1]);
  tSpace[tIdx][1] = 0.0;
  for(int qz=1; qz<((rDim[2]/2)+1); ++qz){

    int qVec[3] = {0, 0, qz};
    int qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);
    tIdx = qz + ((rDim[2]/2)+1)*timePointCounter;

    tSpace[tIdx][0] = 0.5*(qSpace[qIdx][0]*qSpace[qIdx][0] + qSpace[qIdx][1]*qSpace[qIdx][1]);
    tSpace[tIdx][1] = 0.0;
    
    qVec[2] = rDim[2]-qz;
    qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);

    tSpace[tIdx][0] = tSpace[tIdx][0] + 0.5*(qSpace[qIdx][0]*qSpace[qIdx][0] + qSpace[qIdx][1]*qSpace[qIdx][1]);
  }

  if(timePointCounter == (nTimePoints-1)){
    fftw_execute(tSpaceFFT);

    // average over -omega +omega
    for(int qz=0; qz<((rDim[2]/2)+1); ++qz){
      // cant average zero mode
      tIdx = qz;
      DSFFile << qz << "\t" << 0.0 <<"\t" << (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1]) <<"\n";
      for(int omega=1; omega<((nTimePoints/2)+1); ++omega){
        tIdx = qz + ((rDim[2]/2)+1)*omega;
        
          DSFFile << qz << "\t" << omega*freqIntervalSize <<"\t" << (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1]) <<"\n";
      }
      DSFFile << "\n";
    }
  }

  timePointCounter++;

}
