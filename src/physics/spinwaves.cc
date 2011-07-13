#include <cmath>
#include <iostream>
#include <fstream>

#include "globals.h"
#include "spinwaves.h"

void SpinwavesPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * Spinwaves physics module\n");

  PulseDuration = phys["PulseDuration"];
  PulseTotal    = phys["PulseTotal"];
  PulseTemperature = phys["PulseTemperature"];

  for(int i=0; i<3; ++i) {
    FieldStrength[i] = phys["FieldStrength"][i];
  }

  PulseCount = 1;


  lattice.getDimensions(dim[0],dim[1],dim[2]);

  assert( dim[0]*dim[1]*dim[2] > 0 );

  // allocate storage for S_perp projection
  FFTArray = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nspins));

  FFTPlan = fftw_plan_dft_3d(dim[0],dim[1],dim[2],FFTArray,FFTArray,FFTW_FORWARD,FFTW_MEASURE);
  
  std::string filename = "_spw.dat";
  filename = seedname+filename;
  SPWFile.open(filename.c_str());
  SPWFile << "# t [s]\tk=0\tk!=0\tM_AF1_x\tM_AF1_y\tM_AF1_z\n";


  const double sampletime = config.lookup("sim.t_out");
  const double runtime = config.lookup("sim.t_run");
  const int outcount = runtime/sampletime;

  initialised = true;
}

SpinwavesPhysics::~SpinwavesPhysics()
{
  if(initialised == true) {
    fftw_destroy_plan(FFTPlan);

    if(FFTArray != NULL) {
      fftw_free(FFTArray);
    }
    FFTArray = NULL;
  }

  SPWFile.close();
}

void SpinwavesPhysics::run(double realtime, const double dt) {
}

void SpinwavesPhysics::monitor(double realtime, const double dt) {
  using namespace globals;

  double eqtime = config.lookup("sim.t_eq");

  
  if( (realtime > eqtime) && (realtime-eqtime) > (PulseDuration*PulseCount)){
    PulseCount++;
  }

  if( (realtime > eqtime) && ((realtime-eqtime) < (PulseDuration*(PulseTotal)))){
    if(PulseCount%2 == 0) {
      for(int i=0; i<3; ++i) {
        globals::h_app[i] = -FieldStrength[i];
        globals::globalTemperature = PulseTemperature;
      }
    }else{
      for(int i=0; i<3; ++i) {
        globals::h_app[i] = FieldStrength[i];
        globals::globalTemperature = PulseTemperature;
      }
    }
  }else{
    for(int i=0; i<3; ++i) {
      globals::h_app[i] = 0.0;                                                                     
        globals::globalTemperature = config.lookup("sim.temperature");
    }
  }

  
  // calculate structure factor <S^{k}(-q)S^{k}(q)>

  for(int i=0; i<nspins; ++i){
    FFTArray[i][0] = s(i,0);
    FFTArray[i][1] = s(i,1);
  }


  fftw_execute(FFTPlan);

  FFTArray[0][0] = 2.0*FFTArray[0][0];
  FFTArray[0][1] = 2.0*FFTArray[0][1];

  

  // normalise by number of spins and total mode power
  double pow_norm = 0.0;
  for(int i=0; i<dim[0]; ++i){
    for(int j=0; j<dim[1]; ++j){
      for(int k=0; k<dim[2]; ++k){
        const int kVec[3] = {i,j,k};
        const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
        FFTArray[idx][0] = FFTArray[idx][0]/sqrt(dim[0]*dim[1]*dim[2]);
        FFTArray[idx][1] = FFTArray[idx][1]/sqrt(dim[0]*dim[1]*dim[2]);
      }
    }
  }
  
  const double kzero = (FFTArray[0][0]*FFTArray[0][0] + FFTArray[0][1]*FFTArray[0][1]);

  SPWFile << 0 << "\t" << kzero << "\n";

   for(int k=1; k<(dim[2]/2)+1; ++k){
     const int qVec[3]      = {0,0,k};
     const int qVecMinus[3] = {0,0,(dim[2]-k)};
     const int qIdx      = qVec[2] + dim[2]*(qVec[1] + dim[1]*qVec[0]);
     const int qIdxMinus = qVecMinus[2] + dim[2]*(qVecMinus[1] + dim[1]*qVecMinus[0]);
     double Sq      = (FFTArray[qIdx][0]*FFTArray[qIdx][0] + FFTArray[qIdx][1]*FFTArray[qIdx][1]);
     double SqMinus = (FFTArray[qIdxMinus][0]*FFTArray[qIdxMinus][0] + FFTArray[qIdxMinus][1]*FFTArray[qIdxMinus][1]);
     SPWFile << k << "\t" << (Sq*SqMinus) << "\n";
   }
  SPWFile << "\n\n";

}
