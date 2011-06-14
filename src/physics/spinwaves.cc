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
      s(i,2) = -1.0;
    }
  }

  // read from config if spinstates should be dumped
  // (default is false)
  spinDump = false;
  phys.lookupValue("spindump",spinDump);

  // open spin dump file as binary file
  filename = "_spd.dat";
  filename = seedname+filename;
  if(spinDump == true){
    SPDFile.open(filename.c_str(),std::ofstream::binary);
  }

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
  
//  for(int i=0; i<nspins; ++i){
//    FFTArray[i][0] = s(i,0);
//    FFTArray[i][1] = s(i,1);
//  }


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


  double phi = acos(magA[2]*normA);
  double theta   = atan2(magA[0],magA[1]);

  // calculate rotation matrix for rotating m->m_z
  double c_t = cos(theta);
  double c_p = cos(phi);
  double s_t = sin(theta);
  double s_p = sin(phi);

  double rotA[3][3];

  // R_z(phi)*R_y(theta)
//   rot[0][0]=c_t*c_p;
//   rot[0][1]=-s_p;
//   rot[0][2]=c_p*s_t;
//   rot[1][0]=s_p*c_t;
//   rot[1][1]=c_p;
//   rot[1][2]=s_t*s_p;
//   rot[2][0]=-s_t;
//   rot[2][1]=0.0;
//   rot[2][2]=c_t;

  rotA[0][0]=c_t*c_p;
  rotA[0][1]=s_t*c_p;
  rotA[0][2]=-s_p;
  rotA[1][0]=-s_t;
  rotA[1][1]=c_t;
  rotA[1][2]=0.0;
  rotA[2][0]=c_t*s_p;
  rotA[2][1]=s_t*s_p;
  rotA[2][2]=c_p;

  phi = acos(magB[2]*normB);
  theta   = atan2(magB[0],magB[1]);

  // calculate rotation matrix for rotating m->m_z
  c_t = cos(theta);
  c_p = cos(phi);
  s_t = sin(theta);
  s_p = sin(phi);

  double rotB[3][3];
//   rotB[0][0]=c_t*c_p;
//   rotB[0][1]=s_t*c_p;
//   rotB[0][2]=-s_p;
//   rotB[1][0]=-s_t;
//   rotB[1][1]=c_t;
//   rotB[1][2]=0.0;
//   rotB[2][0]=c_t*s_p;
//   rotB[2][1]=s_t*s_p;
//   rotB[2][2]=c_p;

//   rotB[0][0]=c_t*c_p;
//   rotB[0][1]=-s_t;
//   rotB[0][2]=c_t*s_p;
//   rotB[1][0]=s_t*c_p;
//   rotB[1][1]=c_t;
//   rotB[1][2]=s_t*s_p;
//   rotB[2][0]=-s_p;
//   rotB[2][1]=0.0;
//   rotB[2][2]=c_p;

  for(int i=0; i<nspins; ++i){

    if(typeOverride[i] == 0){
      FFTArray[i][0] = rotA[0][0]*s(i,0) + rotA[0][1]*s(i,1) + rotA[0][2]*s(i,2);
      FFTArray[i][1] = rotA[1][0]*s(i,0) + rotA[1][1]*s(i,1) + rotA[1][2]*s(i,2);
    } else {
      FFTArray[i][0] = (rotA[0][0]*(-s(i,0)) + rotA[0][1]*s(i,1) + rotA[0][2]*(-s(i,2)));
      FFTArray[i][1] = (rotA[1][0]*(-s(i,0)) + rotA[1][1]*s(i,1) + rotA[1][2]*(-s(i,2)));
    }
  }

  if(spinDump == true){
    // write spin dump data in binary
    SPDFile.write((char*)FFTArray,nspins*sizeof(fftw_complex));
  }

  fftw_execute(FFTPlan);

  // normalise by number of spins and total mode power
  double pow_norm = 0.0;
  for(int i=0; i<(dim[0]/2)+1; ++i){
    for(int j=0; j<(dim[1]/2)+1; ++j){
      for(int k=0; k<(dim[2]/2)+1; ++k){
        const int kVec[3] = {i,j,k};
        const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
        pow_norm += (FFTArray[idx][0]*FFTArray[idx][0] + FFTArray[idx][1]*FFTArray[idx][1]);
      }
    }
  }
  
  const double kzero = (FFTArray[0][0]*FFTArray[0][0] + FFTArray[0][1]*FFTArray[0][1])/(pow_norm);

  SPWFile << realtime << "\t" << kzero << "\t" << 1.0-kzero;
  SPWFile << "\t"<< magA[0] << "\t" << magA[1] << "\t" << magA[2];
  SPWFile << "\t"<< magB[0] << "\t" << magB[1] << "\t" << magB[2];

   for(int k=1; k<(dim[2]/2)+1; ++k){
     const int kVec[3] = {0,0,k};
     const int idx = kVec[2] + dim[2]*(kVec[1] + dim[1]*kVec[0]);
     double pow = (FFTArray[idx][0]*FFTArray[idx][0] + FFTArray[idx][1]*FFTArray[idx][1]);
     SPWFile << "\t" << pow/pow_norm;
   }
  SPWFile << "\n";

  fftw_execute(FFTPlan);
}
