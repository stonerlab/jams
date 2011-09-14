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



  output.write("  * Initialising FFTW variables...\n");

// --------------------------------------------------------------------------------------------------------------------
// Real space to reciprocal (q) space transform
// --------------------------------------------------------------------------------------------------------------------
  output.write("  * qSpace allocating %f MB\n", (sizeof(fftw_complex)*rDim[0]*rDim[1]*rDim[2])/(1024.0*1024.0));
  qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*rDim[0]*rDim[1]*rDim[2]));
  if(qSpace == NULL){
    jams_error("Failed to allocate qSpace FFT array");
  }
  qSpaceFFT = fftw_plan_dft_3d(rDim[0],rDim[1],rDim[2],qSpace,qSpace,FFTW_FORWARD,FFTW_MEASURE);

// --------------------------------------------------------------------------------------------------------------------
// Space upsampling transform
// --------------------------------------------------------------------------------------------------------------------

  upFac[0] = 1;
  upFac[1] = 1;
  upFac[2] = 1;
  
  output.write("  * qSpace upsampling factors: (%d,%d,%d)\n",upFac[0],upFac[1],upFac[2]);


  for(int i=0;i<3;++i){
    upDim[i] = upFac[i]*rDim[i];
  }
  
  upSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*upDim[0]*upDim[1]*upDim[2]));
  if(upSpace == NULL){
    jams_error("Failed to allocate qSpace FFT array");
  }
  
  const int qzPoints = (upDim[2]/2)+1;

  upSpaceFFT = fftw_plan_dft_3d(upDim[0],upDim[1],upDim[2],upSpace,upSpace,FFTW_FORWARD,FFTW_MEASURE);
  invUpSpaceFFT = fftw_plan_dft_3d(upDim[0],upDim[1],upDim[2],upSpace,upSpace,FFTW_BACKWARD,FFTW_MEASURE);

// --------------------------------------------------------------------------------------------------------------------
// Time to frequency space transform
// --------------------------------------------------------------------------------------------------------------------
//   output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*nspins)/(1024.0*1024.0));
  output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*qzPoints/(1024.0*1024.0)));
//   tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*nspins));
  tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*qzPoints));
  if(qSpace == NULL){
    jams_error("Failed to allocate tSpace FFT array");
  }

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

  const int qzPoints = (upDim[2]/2)+1;

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

  for(int i=0; i<(upDim[0]*upDim[1]*upDim[2]); ++i){
    upSpace[i][0] = 0.0;
    upSpace[i][1] = 0.0;
  }

  for(int i=0;i<rDim[0];++i){
    for(int j=0;j<rDim[1];++j){
      for(int k=0;k<rDim[2];++k){
        int rIdx = k + rDim[2]*(j + rDim[1]*i);
        int upIdx = (k*upFac[2]) + upDim[2]*((j*upFac[1]) + upDim[1]*(i*upFac[0]));
        upSpace[upIdx][0] = qSpace[rIdx][0];
        upSpace[upIdx][1] = qSpace[rIdx][1];
      }
    }
  }

//   fftw_execute(upSpaceFFT);
// 
//   for(int i=0;i<upDim[0];++i){
//     for(int j=0;j<upDim[1];++j){
//       for(int k=((rDim[2]/2)+1);k<(upDim[2]-(rDim[2]/2));++k){
//         int upIdx = k + upDim[2]*(j + upDim[1]*i);
//         upSpace[upIdx][0] = 0.0;
//         upSpace[upIdx][1] = 0.0;
//       }
//     }
//   }
// 
//   
//     
//   fftw_execute(invUpSpaceFFT);

  for(int i=0; i<(upDim[0]*upDim[1]*upDim[2]); ++i){
    upSpace[i][0] /= (upDim[0]*upDim[1]*upDim[2]);
    upSpace[i][1] /= (upDim[0]*upDim[1]*upDim[2]);
  }
  //fftw_execute(upSpaceFFT);

  tSpace[tIdx][0] = upSpace[0][0];
  tSpace[tIdx][1] = upSpace[0][1];
  for(int qz=1; qz<qzPoints; ++qz){

    int qVec[3] = {0, 0, qz};
    int qIdx = qVec[2] + upDim[2]*(qVec[1] + upDim[1]*qVec[0]);
    assert(qIdx < nspins); assert(qIdx > -1);
    tIdx = qz + qzPoints*timePointCounter;
    assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);

    tSpace[tIdx][0] = 0.5*upSpace[qIdx][0];
    tSpace[tIdx][1] = 0.5*upSpace[qIdx][1];
    
    qVec[2] = upDim[2]-qz;
    qIdx = qVec[2] + upDim[2]*(qVec[1] + upDim[1]*qVec[0]);
    assert(qIdx < nspins); assert(qIdx > -1);

    tSpace[tIdx][0] = tSpace[tIdx][0] + 0.5*(upSpace[qIdx][0]);
    tSpace[tIdx][1] = tSpace[tIdx][1] + 0.5*(upSpace[qIdx][1]);
  }

  if(timePointCounter == (nTimePoints-1)){

    // apply windowing function

    for(int t=0; t<nTimePoints;++t){
      for(int qz=0; qz<qzPoints; ++qz){
        tIdx = qz + qzPoints*t;
        assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);
        tSpace[tIdx][0] = tSpace[tIdx][0]*FFTWindow(t,nTimePoints,HAMMING);
        tSpace[tIdx][1] = tSpace[tIdx][1]*FFTWindow(t,nTimePoints,HAMMING);
      }
    }
    
    fftw_execute(tSpaceFFT);


    // normalise transform and apply symmetry -omega omega
    for(int t=0; t<(nTimePoints/2)+1;++t){
      for(int qz=0; qz<qzPoints; ++qz){
        tIdx = qz + qzPoints*t;
        const int tIdxMinus = qz + qzPoints*(nTimePoints - t);
        assert(tIdx < qzPoints*nTimePoints); assert(tIdx > -1);
        assert(tIdxMinus < qzPoints*nTimePoints); assert(tIdxMinus > -1);

        tSpace[tIdx][0] = 0.5*(tSpace[tIdx][0] + tSpace[tIdxMinus][0])/sqrt(double(nspins)*double(nTimePoints));
        tSpace[tIdx][1] = 0.5*(tSpace[tIdx][1] + tSpace[tIdxMinus][1])/sqrt(double(nspins)*double(nTimePoints));

        // zero -omega to avoid accidental use
        tSpace[tIdxMinus][0] = 0.0; tSpace[tIdxMinus][1] = 0.0;
      }
    }

    // apply symmetry -omega omega

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
        double sigma = 0.05;
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
        tIdx = qz + qzPoints*omega;
        DSFFile << qz << "\t" << omega*freqIntervalSize <<"\t" << (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1]) <<"\t";
        DSFFile << (smoothData[omega][0]*smoothData[omega][0] + smoothData[omega][1]*smoothData[omega][1])/convolutionNorm << "\n";
      }
      DSFFile << "\n";
    }
  }
  
  timePointCounter++;

}

double DynamicSFPhysics::FFTWindow(const int n, const int nTotal, const FFTWindowType type){
  switch(type)
  {
    case GAUSSIAN:
      // sigma = 0.4
      return (1.0/(sqrt(2.0*M_PI)*0.4))*exp(-( ((double(n)/double(nTotal-1))-0.5) * (double(n)/double(nTotal-1))-0.5  )
                /(2.0*0.16));
      break;
    case HAMMING:
      return 0.54 - 0.46*cos((2.0*M_PI*n)/double(nTotal-1));
      break;
    default:
      // default to Hamming window
      return 0.54 - 0.46*cos((2.0*M_PI*n)/double(nTotal-1));
      break;
  }
}
