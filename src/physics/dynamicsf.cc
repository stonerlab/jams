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
  output.write("  * Time sample points: %d\n",nTimePoints);

  freqIntervalSize = (2.0*M_PI)/(sampletime*nTimePoints);
  output.write("  * Sample frequency: %f [GHz]\n",freqIntervalSize/1E9);

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

  const int qzPoints = (rDim[2]/2)+1;


  output.write("  * Allocating FFTW arrays...\n");
  output.write("  * qSpace allocating %f MB\n", (sizeof(fftw_complex)*rDim[0]*rDim[1]*rDim[2])/(1024.0*1024.0));
  qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*rDim[0]*rDim[1]*rDim[2]));
  if(qSpace == NULL){
    jams_error("Failed to allocate qSpace FFT array");
  }
//   output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*nspins)/(1024.0*1024.0));
  output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*qzPoints/(1024.0*1024.0)));
//   tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*nspins));
  tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*qzPoints));
  if(qSpace == NULL){
    jams_error("Failed to allocate tSpace FFT array");
  }

  output.write("  * Planning FFTW transform...\n");
  qSpaceFFT = fftw_plan_dft_3d(rDim[0],rDim[1],rDim[2],qSpace,qSpace,FFTW_FORWARD,FFTW_MEASURE);
  
// --------------------------------------------------------------------------------------------------------------------
// FFTW Advanced interface for time transform
// --------------------------------------------------------------------------------------------------------------------
  int rank       = 1;
  int sizeN[]   = {nTimePoints};
  int howmany    = qzPoints;
  int inembed[] = {nTimePoints};  int onembed[] = {nTimePoints};
  int istride    = qzPoints;      int ostride    = qzPoints;
  int idist      = 1;             int odist      = 1;

  tSpaceFFT = fftw_plan_many_dft(rank,sizeN,howmany,tSpace,inembed,istride,idist,tSpace,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
// --------------------------------------------------------------------------------------------------------------------
  
//   int tDim[4] = {nTimePoints,rDim[0],rDim[1],rDim[2]};
//   tSpaceFFT = fftw_plan_dft(4,tDim,tSpace,tSpace,FFTW_FORWARD,FFTW_ESTIMATE);

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

  const int qzPoints = (rDim[2]/2)+1;

  // ASSUMPTION: Spin array is C-ordered in space
//   if(componentImag == -1){
//     for(int i=0; i<nspins; ++i){
//       qSpace[i][0] = s(i,componentReal);
//       qSpace[i][1] = 0.0;
//     }
//   } else {
//     for(int i=0; i<nspins; ++i){
//       qSpace[i][0] = s(i,componentReal);
//       qSpace[i][1] = s(i,componentImag);
//     }
//   }
// 
//   fftw_execute(qSpaceFFT);
// 
//   // average over -q +q
//   
//   // cant average zero mode
//   int tIdx = ((rDim[2]/2)+1)*timePointCounter;
//   tSpace[tIdx][0] = qSpace[0][0];
//   tSpace[tIdx][1] = qSpace[0][1];
//   for(int qz=1; qz<((rDim[2]/2)+1); ++qz){
// 
//     int qVec[3] = {0, 0, qz};
//     int qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);
//     tIdx = qz + ((rDim[2]/2)+1)*timePointCounter;
// 
//     tSpace[tIdx][0] = 0.5*qSpace[qIdx][0];
//     tSpace[tIdx][1] = 0.5*qSpace[qIdx][1];
//     
//     qVec[2] = rDim[2]-qz;
//     qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);
// 
//     tSpace[tIdx][0] = tSpace[tIdx][0] + 0.5*(qSpace[qIdx][0]);
//     tSpace[tIdx][1] = tSpace[tIdx][1] + 0.5*(qSpace[qIdx][1]);
//   }
// 
//   if(timePointCounter == (nTimePoints-1)){
//     fftw_execute(tSpaceFFT);
// 
//     // average over -omega +omega
//     for(int qz=0; qz<((rDim[2]/2)+1); ++qz){
//       // cant average zero mode
//       tIdx = qz;
//       DSFFile << qz << "\t" << 0.0 <<"\t" << (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1]) <<"\n";
//       for(int omega=1; omega<((nTimePoints/2)+1); ++omega){
//         tIdx = qz + ((rDim[2]/2)+1)*omega;
//         
//           DSFFile << qz << "\t" << omega*freqIntervalSize <<"\t" << (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1]) <<"\n";
//       }
//       DSFFile << "\n";
//     }
//   }

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
  int tIdx = qzPoints*timePointCounter;
  assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);

  tSpace[tIdx][0] = qSpace[0][0];
  tSpace[tIdx][1] = qSpace[0][1];
  for(int qz=1; qz<qzPoints; ++qz){

    int qVec[3] = {0, 0, qz};
    int qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);
    assert(qIdx < nspins); assert(qIdx > -1);
    tIdx = qz + qzPoints*timePointCounter;
    assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);

    tSpace[tIdx][0] = 0.5*qSpace[qIdx][0];
    tSpace[tIdx][1] = 0.5*qSpace[qIdx][1];
    
    qVec[2] = rDim[2]-qz;
    qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);
    assert(qIdx < nspins); assert(qIdx > -1);

    tSpace[tIdx][0] = tSpace[tIdx][0] + 0.5*(qSpace[qIdx][0]);
    tSpace[tIdx][1] = tSpace[tIdx][1] + 0.5*(qSpace[qIdx][1]);
  }

  if(timePointCounter == (nTimePoints-1)){

    // apply windowing function

    for(int t=0; t<nTimePoints;++t){
//       // Gaussian
//       const double sigma = 0.4;
//       const double x = (double(t)/double(nTimePoints-1))-0.5;
//       const double window = (1.0/(sqrt(2.0*M_PI)*sigma))*exp(-(x*x)/(2.0*sigma*sigma));
      // Hamming
      const double window = 0.54 - 0.46*cos((2.0*M_PI*t)/double(nTimePoints-1));
      std::cerr<<t<<"\t"<<window<<std::endl;
      for(int qz=0; qz<qzPoints; ++qz){
        tIdx = qz + qzPoints*t;
        assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);
        tSpace[tIdx][0] = tSpace[tIdx][0]*window;
        tSpace[tIdx][1] = tSpace[tIdx][1]*window;
      }
    }
    
    fftw_execute(tSpaceFFT);


    for(int t=0; t<nTimePoints;++t){
      for(int qz=0; qz<qzPoints; ++qz){
        tIdx = qz + qzPoints*t;
        assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);
        tSpace[tIdx][0] = tSpace[tIdx][0]/sqrt(double(nspins)*double(nTimePoints));
        tSpace[tIdx][1] = tSpace[tIdx][1]/sqrt(double(nspins)*double(nTimePoints));
      }
    }

    fftw_complex* gaussian = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*((nTimePoints/2)+1)));
    fftw_complex* smoothData = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*((nTimePoints/2)+1)));
    
    fftw_plan gaussianFFT  = fftw_plan_dft_1d((nTimePoints/2)+1,gaussian,gaussian,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_plan smoothDataFFT  = fftw_plan_dft_1d((nTimePoints/2)+1,smoothData,smoothData,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_plan convolutionFFT  = fftw_plan_dft_1d((nTimePoints/2)+1,smoothData,smoothData,FFTW_BACKWARD,FFTW_ESTIMATE);

    
    

    // average over -omega +omega
    for(int qz=0; qz<qzPoints; ++qz){
      
      // perform Gaussian convolution
      for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
        int tIdx = qz+qzPoints*omega;
        double sigma = 0.2;
//         double x = (double(omega)/((nTimePoints/2)+1) - 0.5);
        double x = (double(omega)/((nTimePoints/2)+1) );
        gaussian[omega][0] = (1.0/(sqrt(2.0*M_PI)*sigma))*exp(-(x*x)/(2.0*sigma*sigma));
        gaussian[omega][1] = 0.0;
        smoothData[omega][0] = tSpace[tIdx][0];
        smoothData[omega][1] = tSpace[tIdx][1];

      }
      fftw_execute(gaussianFFT);
      fftw_execute(smoothDataFFT);

      for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
        double a = gaussian[omega][0]; double b = gaussian[omega][1]; 
        double c = (smoothData[omega][0]); double d = (smoothData[omega][1]); 
        smoothData[omega][0] = (a*c - b*d)/sqrt(double(((nTimePoints/2)+1)));
        smoothData[omega][1] = (a*d + b*c)/sqrt(double(((nTimePoints/2)+1)));
      }
      fftw_execute(convolutionFFT);

      double convolutionNorm = 0.0;
      // normalize convoluted data area = 1
//       for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
//         convolutionNorm = convolutionNorm + smoothData[omega][0]*smoothData[omega][0]+smoothData[omega][1]*smoothData[omega][1];
//       }
      // normalize convoluted data max = 1
      for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
        if(smoothData[omega][0]*smoothData[omega][0]+smoothData[omega][1]*smoothData[omega][1] > convolutionNorm){
          convolutionNorm = (smoothData[omega][0]*smoothData[omega][0]+smoothData[omega][1]*smoothData[omega][1]);
        }
      }



      for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
        int qVec[3] = {0, 0, qz};
        tIdx = qz + qzPoints*omega;
        DSFFile << qz << "\t" << omega*freqIntervalSize <<"\t" << (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1]) <<"\t";
        DSFFile << (smoothData[omega][0]*smoothData[omega][0] + smoothData[omega][1]*smoothData[omega][1])/convolutionNorm << "\n";
      }
      DSFFile << "\n";
    }
  }
  
  timePointCounter++;

}
