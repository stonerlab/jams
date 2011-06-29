#include <cmath>
#include <iostream>
#include <fstream>

#include "globals.h"
#include "spinwaves.h"

void SpinwavesPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  output.write("  * Spinwaves physics module\n");

  lattice.getDimensions(dim[0],dim[1],dim[2]);

  assert( dim[0]*dim[1]*dim[2] > 0 );

  // allocate storage for S_perp projection
  FFTArray = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nspins));

  FFTPlan = fftw_plan_dft_3d(dim[0],dim[1],dim[2],FFTArray,FFTArray,FFTW_FORWARD,FFTW_MEASURE);
  
  std::string filename = "_spw.dat";
  filename = seedname+filename;
  SPWFile.open(filename.c_str());
  SPWFile << "# t [s]\tk=0\tk!=0\tM_AF1_x\tM_AF1_y\tM_AF1_z\n";


  typeOverride.resize(nspins);

  int count=0;
  int countAF1=0;
  int countAF2=0;
  for(int i=0; i<dim[0]; ++i){
    for(int j=0; j<dim[1]; ++j){
      for(int k=0; k<dim[2]; ++k){
        if(i%2 == 0){
          if(j%2 == 0){
            if(k%2 == 0){
              typeOverride[count]=0;
              countAF1++;
              count++;
            }else{
              typeOverride[count]=1;
              countAF2++;
              count++;
            }
          }else{
            if(k%2 == 0){
              typeOverride[count]=1;
              countAF2++;
              count++;
            }else{
              typeOverride[count]=0;
              countAF1++;
              count++;
            }
          }
        }else{
          if(j%2 == 0){
            if(k%2 == 0){
              typeOverride[count]=1;
              countAF2++;
              count++;
            }else{
              typeOverride[count]=0;
              countAF1++;
              count++;
            }
          }else{
            if(k%2 == 0){
              typeOverride[count]=0;
              countAF1++;
              count++;
            }else{
              typeOverride[count]=1;
              countAF2++;
              count++;
            }
          }
        }
      }
    }
  }

  output.write("AF counts AF1:%d AF2:%d\n",countAF1,countAF2);

  for(int i=0;i<nspins;++i){
    if(typeOverride[i] == 0){
      s(i,0) = 0.0;
      s(i,1) = 0.0;
      s(i,2) = 1.0;
    }else{
      s(i,0) = 0.0;
      s(i,1) = 0.0;
      s(i,2) = 1.0;
    }
  }

  // read from config if spinstates should be dumped
  // (default is false)
  //spinDump = false;
  //phys.lookupValue("spindump",spinDump);

  // open spin dump file as binary file
  filename = "_spd.dat";
  filename = seedname+filename;
  //if(spinDump == true){
  //  SPDFile.open(filename.c_str(),std::ofstream::binary);
  //}

  const double sampletime = config.lookup("sim.t_out");
  const double runtime = config.lookup("sim.t_run");
  const int outcount = runtime/sampletime;

  SPDFile.write((char*)&outcount,sizeof(int));
  SPDFile.write((char*)&dim[0],3*sizeof(int));
  SPDFile.write((char*)&sampletime,sizeof(double));

  for(int i=0;i<nspins;++i){
    int type = lattice.getType(i);
    SPDFile.write((char*)&type,sizeof(int));
  }

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
  
  // calculate structure factor <S^{k}(-q)S^{k}(q)>
  double magA[3] = {0.0,0.0,0.0};
  double magB[3] = {0.0,0.0,0.0};
  
  for(int i=0; i<nspins; ++i){
    if(typeOverride[i] == 0){
      magA[0] += s(i,0);
      magA[1] += s(i,1);
      magA[2] += s(i,2);
    }else{
      magB[0] += s(i,0);
      magB[1] += s(i,1);
      magB[2] += s(i,2);
    }
  }

  double normA = 1.0/sqrt(magA[0]*magA[0]+magA[1]*magA[1]+magA[2]*magA[2]);
  double normB = 1.0/sqrt(magB[0]*magB[0]+magB[1]*magB[1]+magB[2]*magB[2]);

  for(int j=0; j<3; ++j){
    magA[j] = magA[j]/(0.5*nspins);
    magB[j] = magB[j]/(0.5*nspins);
  }

  for(int i=0; i<nspins; ++i){

    if(typeOverride[i] == 0){
      FFTArray[i][0] = s(i,0);
      FFTArray[i][1] = s(i,1);
    } else {
      FFTArray[i][0] = -s(i,0);
//       FFTArray[i][0] = s(i,0);
      FFTArray[i][1] = s(i,1);
    }
  }

//   if(spinDump == true){
//     // write spin dump data in binary
//     SPDFile.write((char*)FFTArray,nspins*sizeof(fftw_complex));
//   }

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
