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

  // read spin type cofactors (i.e. for Holstein Primakoff
  // transformations and the like)
  
  libconfig::Setting &mat = config.lookup("materials");
  coFactors.resize(lattice.numTypes(),3);
  for(int i=0; i<lattice.numTypes(); ++i){
    for(int j=0; j<3; ++j){
      coFactors(i,j) = mat[i]["coFactors"][j];
    }
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

  // window time
  if( config.exists("physics.t_window") == true) {
    t_window = phys["t_window"];
  }else{
    t_window = config.lookup("sim.t_run");
  }
  const double dt = config.lookup("sim.t_step");
  const double t_out = config.lookup("sim.t_out");
  steps_window = static_cast<unsigned long>(t_window/dt);
  output.write("  * Window time: %1.8e (%lu steps)\n",t_window,steps_window);
  steps_window = t_window/(t_out);
  if(nTimePoints%(steps_window) != 0){
    jams_error("Window time must be an integer multiple of the run time");
  }

  freqIntervalSize = (2.0*M_PI)/(sampletime*steps_window);
  output.write("  * Sample frequency: %f [GHz]\n",freqIntervalSize/1E9);


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

  
//   int tDim[4] = {nTimePoints,rDim[0],rDim[1],rDim[2]};
//   tSpaceFFT = fftw_plan_dft(4,tDim,tSpace,tSpace,FFTW_FORWARD,FFTW_ESTIMATE);


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
    if(tSpace != NULL) {
      fftw_free(tSpace);
      tSpace = NULL;
    }
    if(imageSpace != NULL) {
      fftw_free(imageSpace);
      imageSpace = NULL;
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

  const int qzPoints = (upDim[2]/2)+1;

  // ASSUMPTION: Spin array is C-ordered in space

  if(componentImag == -1){

    for(int i=0; i<nspins; ++i){
      int type = lattice.getType(i);

      qSpace[i][0] = coFactors(type,componentReal)*s(i,componentReal);
      qSpace[i][1] = 0.0;
    }
  } else {
    for(int i=0; i<nspins; ++i){
      int type = lattice.getType(i);
      qSpace[i][0] = coFactors(type,componentReal)*s(i,componentReal);
      qSpace[i][1] = coFactors(type,componentImag)*s(i,componentImag);
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
    timeTransform();
    outputImage();


//     // apply symmetry -omega omega
// 
//     fftw_complex* gaussian = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*((nTimePoints/2)+1)));
//     fftw_complex* smoothData = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*((nTimePoints/2)+1)));
//     
//     fftw_plan gaussianFFT  = fftw_plan_dft_1d((nTimePoints/2)+1,gaussian,gaussian,FFTW_FORWARD,FFTW_ESTIMATE);
//     fftw_plan smoothDataFFT  = fftw_plan_dft_1d((nTimePoints/2)+1,smoothData,smoothData,FFTW_FORWARD,FFTW_ESTIMATE);
//     fftw_plan convolutionFFT  = fftw_plan_dft_1d((nTimePoints/2)+1,smoothData,smoothData,FFTW_BACKWARD,FFTW_ESTIMATE);
//     
// 
//     // average over -omega +omega
//     for(int qz=0; qz<qzPoints; ++qz){
//       
//       // perform Gaussian convolution
//       for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
//         int tIdx = qz+qzPoints*omega;
//         double sigma = 0.05;
// //         double x = (double(omega)/((nTimePoints/2)+1) - 0.5);
//         double x = (double(omega)/((nTimePoints/2)+1) );
//         gaussian[omega][0] = (1.0/(sqrt(2.0*M_PI)*sigma))*exp(-(x*x)/(2.0*sigma*sigma));
//         gaussian[omega][1] = 0.0;
//         smoothData[omega][0] = tSpace[tIdx][0];
//         smoothData[omega][1] = tSpace[tIdx][1];
// 
//       }
//       fftw_execute(gaussianFFT);
//       fftw_execute(smoothDataFFT);
// 
//       for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
//         double a = gaussian[omega][0]; double b = gaussian[omega][1]; 
//         double c = (smoothData[omega][0]); double d = (smoothData[omega][1]); 
//         smoothData[omega][0] = (a*c - b*d)/sqrt(double(((nTimePoints/2)+1)));
//         smoothData[omega][1] = (a*d + b*c)/sqrt(double(((nTimePoints/2)+1)));
//       }
//       fftw_execute(convolutionFFT);
// 
//       double convolutionNorm = 0.0;
//       // normalize convoluted data area = 1
// //       for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
// //         convolutionNorm = convolutionNorm + smoothData[omega][0]*smoothData[omega][0]+smoothData[omega][1]*smoothData[omega][1];
// //       }
//       // normalize convoluted data max = 1
//       for(int omega=0; omega<((nTimePoints/2)+1); ++omega){
//         if(smoothData[omega][0]*smoothData[omega][0]+smoothData[omega][1]*smoothData[omega][1] > convolutionNorm){
//           convolutionNorm = (smoothData[omega][0]*smoothData[omega][0]+smoothData[omega][1]*smoothData[omega][1]);
//         }
//       }

  }
  
  timePointCounter++;

}

void DynamicSFPhysics::timeTransform()
{
  using namespace globals;

  const int nTransforms = (nTimePoints/steps_window);
  const double normTransforms = 1.0/double(nTransforms);

  output.write("Performing %d window transforms\n",nTransforms);

  const int qzPoints    = (upDim[2]/2) + 1;
  const int omegaPoints = (steps_window/2) + 1;

  // allocate the image space
  imageSpace = static_cast<double*>(fftw_malloc(sizeof(double) * omegaPoints * qzPoints));
  for(int i=0; i<omegaPoints * qzPoints; ++i){
    imageSpace[i] = 0.0;
  }

  for(int i=0; i<nTransforms; ++i){ // integer division is guaranteed in the initialisation

    const int t0 = i*steps_window;
    const int tEnd = (i+1)*steps_window;

    int rank       = 1;
    int sizeN[]   = {steps_window};
    int howmany    = qzPoints;
    int inembed[] = {steps_window}; int onembed[] = {steps_window};
    int istride    = qzPoints;      int ostride    = qzPoints;
    int idist      = 1;             int odist      = 1;
    fftw_complex* startPtr = (tSpace+i*steps_window*qzPoints); // pointer arithmatic

    fftw_plan tSpaceFFT = fftw_plan_many_dft(rank,sizeN,howmany,startPtr,inembed,istride,idist,startPtr,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
    
    // apply windowing function

    for(int t=0; t<steps_window; ++t){
      for(int qz=0; qz<qzPoints; ++qz){
        const int tIdx = qz + qzPoints*(t+t0);
        tSpace[tIdx][0] = tSpace[tIdx][0]*FFTWindow(t,steps_window,HAMMING);
        tSpace[tIdx][1] = tSpace[tIdx][1]*FFTWindow(t,steps_window,HAMMING);
      }
    }
    
    fftw_execute(tSpaceFFT);


    // normalise transform and apply symmetry -omega omega
    for(int t=0; t<omegaPoints;++t){
      for(int qz=0; qz<qzPoints; ++qz){
        const int tIdx = qz + qzPoints*(t0+t);
        const int tIdxMinus = qz + qzPoints*( (tEnd-1) - t);
        assert( tIdx >= 0 );
        assert( tIdx < (nTimePoints*qzPoints) );
        assert( tIdxMinus >= 0 );
        assert( tIdxMinus < (nTimePoints*qzPoints) );

        tSpace[tIdx][0] = 0.5*(tSpace[tIdx][0] + tSpace[tIdxMinus][0])/sqrt(double(nspins)*double(steps_window));
        tSpace[tIdx][1] = 0.5*(tSpace[tIdx][1] + tSpace[tIdxMinus][1])/sqrt(double(nspins)*double(steps_window));

        // zero -omega to avoid accidental use
        tSpace[tIdxMinus][0] = 0.0; tSpace[tIdxMinus][1] = 0.0;

        // assign pixels to image
        int imageIdx = qz+qzPoints*t;
        assert( imageIdx >= 0 );
        assert( imageIdx < (omegaPoints * qzPoints) );
        imageSpace[imageIdx] = imageSpace[imageIdx] + (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1])*normTransforms;
      }
    }

    startPtr = NULL;
    
    fftw_destroy_plan(tSpaceFFT);
  }
}

void DynamicSFPhysics::outputImage()
{
  using namespace globals;
  std::ofstream DSFFile;

  std::string filename = "_dsf.dat";
  filename = seedname+filename;
  DSFFile.open(filename.c_str());
  const int qzPoints = (upDim[2]/2)+1;
  for(int qz=0; qz<qzPoints; ++qz){
    for(int omega=0; omega<((steps_window/2)+1); ++omega){
      int tIdx = qz + qzPoints*omega;
      DSFFile << qz << "\t" << omega*freqIntervalSize <<"\t" << imageSpace[tIdx] <<"\n";
    }
    DSFFile << std::endl;
  }
  
  DSFFile.close();
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
