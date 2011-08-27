#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "globals.h"
#include "spinwaves.h"

void SpinwavesPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * Spinwaves physics module\n");
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

  if( config.exists("physics.SizeOverride") == true) {
    for(int i=0; i<3; ++i) {
      dim[i] = phys["SizeOverride"][i];
    }
  output.write("  * Lattice size override [%d,%d,%d]\n",dim[0],dim[1],dim[2]);
  }else{
    lattice.getDimensions(dim[0],dim[1],dim[2]);
  }

  assert( dim[0]*dim[1]*dim[2] > 0 );

  // allocate storage for S_perp projection
  FFTArray = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nspins));

  FFTPlan = fftw_plan_dft_3d(dim[0],dim[1],dim[2],FFTArray,FFTArray,FFTW_FORWARD,FFTW_MEASURE);
  
  std::string filename = "_spw.dat";
  filename = seedname+filename;
  SPWFile.open(filename.c_str());
  SPWFile << "# t [s]\tk=0\tk!=0\tM_AF1_x\tM_AF1_y\tM_AF1_z\n";

  filename = "_modes.dat";
  filename = seedname+filename;
  ModeFile.open(filename.c_str());

  //const double sampletime = config.lookup("sim.t_out");
  //const double runtime = config.lookup("sim.t_run");
  //const int outcount = runtime/sampletime;

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
  ModeFile.close();
  SPWFile.close();
  TTMFile.close();
}

void SpinwavesPhysics::run(double realtime, const double dt) {
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

  globalTemperature = electronTemp;
}

void SpinwavesPhysics::monitor(double realtime, const double dt) {
  using namespace globals;

  TTMFile << realtime << "\t" << electronTemp << "\t" << phononTemp << "\t" << pumpTemp << "\n";

//-----------------------------------------------------------------------------
// Perpendicular Structure Factor
//-----------------------------------------------------------------------------
  // calculate structure factor <S^{k}(-q)S^{k}(q)>

  for(int i=0; i<nspins; ++i){
    FFTArray[i][0] = s(i,0);
    FFTArray[i][1] = s(i,1);
  }


  fftw_execute(FFTPlan);


  // normalise by number of spins and total mode power
  double pow_norm = 0.0;
  for(int i=0; i<dim[0]; ++i){
    for(int j=0; j<dim[1]; ++j){
      for(int k=0; k<dim[2]; ++k){
        const int kVec[3] = {i,j,k};
        const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
        pow_norm = pow_norm+sqrt(FFTArray[idx][0]*FFTArray[idx][0]+FFTArray[idx][1]*FFTArray[idx][1]);
      }
    }
  }
  
  pow_norm = (2.0*M_PI)/pow_norm;
  for(int i=0; i<dim[0]; ++i){
    for(int j=0; j<dim[1]; ++j){
      for(int k=0; k<dim[2]; ++k){
        const int kVec[3] = {i,j,k};
        const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
        FFTArray[idx][0]=FFTArray[idx][0]*pow_norm;
        FFTArray[idx][1]=FFTArray[idx][1]*pow_norm;
      }
    }
  }

  std::vector<double> StructureFactorPerp((dim[2]/2)+1);
  
  StructureFactorPerp[0] = (FFTArray[0][0]*FFTArray[0][0] + FFTArray[0][1]*FFTArray[0][1]);
   for(int k=1; k<(dim[2]/2)+1; ++k){
     const int qVec[3]      = {0,0,k};
     const int qVecMinus[3] = {0,0,(dim[2]-k)};
     const int qIdx      = qVec[2] + dim[2]*(qVec[1] + dim[1]*qVec[0]);
     const int qIdxMinus = qVecMinus[2] + dim[2]*(qVecMinus[1] + dim[1]*qVecMinus[0]);
     double Sq      = sqrt(FFTArray[qIdx][0]*FFTArray[qIdx][0] + FFTArray[qIdx][1]*FFTArray[qIdx][1]);
     double SqMinus = sqrt(FFTArray[qIdxMinus][0]*FFTArray[qIdxMinus][0] + FFTArray[qIdxMinus][1]*FFTArray[qIdxMinus][1]);
     StructureFactorPerp[k] = (Sq*SqMinus);
   }
//-----------------------------------------------------------------------------
// Parallel Structure Factor
//-----------------------------------------------------------------------------
  for(int i=0; i<nspins; ++i){
    FFTArray[i][0] = s(i,2);
    FFTArray[i][1] = 0.0;
  }


  fftw_execute(FFTPlan);

  // normalise by number of spins and total mode power
  pow_norm = 0.0;
  for(int i=0; i<dim[0]; ++i){
    for(int j=0; j<dim[1]; ++j){
      for(int k=0; k<dim[2]; ++k){
        const int kVec[3] = {i,j,k};
        const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
        pow_norm = pow_norm+sqrt(FFTArray[idx][0]*FFTArray[idx][0]+FFTArray[idx][1]*FFTArray[idx][1]);
      }
    }
  }
  
  pow_norm = (2.0*M_PI)/pow_norm;
  for(int i=0; i<dim[0]; ++i){
    for(int j=0; j<dim[1]; ++j){
      for(int k=0; k<dim[2]; ++k){
        const int kVec[3] = {i,j,k};
        const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
        FFTArray[idx][0]=FFTArray[idx][0]*pow_norm;
        FFTArray[idx][1]=FFTArray[idx][1]*pow_norm;
      }
    }
  }

  std::vector<double> StructureFactorPara((dim[2]/2)+1);
  
  StructureFactorPara[0] = (FFTArray[0][0]*FFTArray[0][0] + FFTArray[0][1]*FFTArray[0][1]);
   for(int k=1; k<(dim[2]/2)+1; ++k){
     const int qVec[3]      = {0,0,k};
     const int qVecMinus[3] = {0,0,(dim[2]-k)};
     const int qIdx      = qVec[2] + dim[2]*(qVec[1] + dim[1]*qVec[0]);
     const int qIdxMinus = qVecMinus[2] + dim[2]*(qVecMinus[1] + dim[1]*qVecMinus[0]);
     double Sq      = sqrt(FFTArray[qIdx][0]*FFTArray[qIdx][0] + FFTArray[qIdx][1]*FFTArray[qIdx][1]);
     double SqMinus = sqrt(FFTArray[qIdxMinus][0]*FFTArray[qIdxMinus][0] + FFTArray[qIdxMinus][1]*FFTArray[qIdxMinus][1]);
     StructureFactorPara[k] = (Sq*SqMinus);
   }

//-----------------------------------------------------------------------------
// Print Files
//-----------------------------------------------------------------------------

   for(int k=0; k<(dim[2]/2)+1; ++k){
     SPWFile << k << "\t" << StructureFactorPerp[k] << "\t" << StructureFactorPara[k] << "\n";
   }
  SPWFile << "\n\n";

  ModeFile << realtime;
  for(int k=0; k<(dim[2]/2)+1;++k){
    ModeFile << "\t" << StructureFactorPerp[k];
  }
  for(int k=0; k<(dim[2]/2)+1;++k){
    ModeFile << "\t" << StructureFactorPara[k];
  }
  ModeFile << "\n";

}
