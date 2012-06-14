#include "globals.h"
#include "dynamicsf.h"

#include <fftw3.h>
#include <string>
#include <cstring>
#include <map>

void DynamicSFPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  const double sampletime = config.lookup("sim.t_out");
  const double runtime = config.lookup("sim.t_run");
  nTimePoints = runtime/sampletime;

  output.write("WARNING: General DSF only works for CUBIC systems");

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
    


    lattice.getKspaceDimensions(rDim[0],rDim[1],rDim[2]);
    output.write("  * Kspace Size [%d,%d,%d]\n",rDim[0],rDim[1],rDim[2]);

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

  freqIntervalSize = (M_PI)/(sampletime*steps_window);
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


//---------------------------------------------------------------------------------
//  Create map from spin number to col major index of qSpace array
//---------------------------------------------------------------------------------

    // find min and max coordinates
    int xmin,xmax;
    int ymin,ymax;
    int zmin,zmax;

    // populate initial values
    lattice.getSpinIntCoord(0,xmin,ymin,zmin);
    lattice.getSpinIntCoord(0,xmax,ymax,zmax);

    for(int i=0; i<nspins; ++i){
        int x,y,z;
        lattice.getSpinIntCoord(i,x,y,z);
        if(x < xmin){ xmin = x; }
        if(x > xmax){ xmax = x; }

        if(y < ymin){ ymin = y; }
        if(y > ymax){ ymax = y; }

        if(z < zmin){ zmin = z; }
        if(z > zmax){ zmax = z; }
    }

  output.write("  * qSpace range: [ %d:%d , %d:%d , %d:%d ]\n", xmin, xmax, ymin, ymax, zmin, zmax);

    spinToKspaceMap.resize(nspins);
    for(int i=0; i<nspins; ++i){
        int x,y,z;
        lattice.getSpinIntCoord(i,x,y,z);


        int idx = (x*rDim[1]+y)*rDim[2]+z;
        spinToKspaceMap[i] = idx;
    }


// --------------------------------------------------------------------------------------------------------------------
// Time to frequency space transform
// --------------------------------------------------------------------------------------------------------------------
  output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*rDim[2]/(1024.0*1024.0)));
  tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*rDim[2]));
  if(qSpace == NULL){
    jams_error("Failed to allocate tSpace FFT array");
  }

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


    // zero qspace array
    memset(qSpace, 0, rDim[0]*rDim[1]*rDim[2]*sizeof(fftw_complex));

  if(componentImag == -1){
    std::vector<double> mag(lattice.numTypes(),0.0);
 
    for(int i=0; i<nspins; ++i){
      const int type = lattice.getType(i);
      mag[type] += coFactors(type,componentReal)*s(i,componentReal);
    }
 
    for(int t=0; t<lattice.numTypes(); ++t){
      mag[t] = mag[t]/static_cast<double>(lattice.getTypeCount(t));
    }
 

    for(int i=0; i<nspins; ++i){
      const int type = lattice.getType(i);
      const int idx = spinToKspaceMap[i];
 
      qSpace[idx][0] = coFactors(type,componentReal)*s(i,componentReal)
                      -mag[lattice.getType(i)];
      qSpace[idx][1] = 0.0;
    }
  } else {
    for(int i=0; i<nspins; ++i){
      int type = lattice.getType(i);
      const int idx = spinToKspaceMap[i];
      qSpace[idx][0] = coFactors(type,componentReal)*s(i,componentReal);
      qSpace[idx][1] = coFactors(type,componentImag)*s(i,componentImag);
    }
  }

  fftw_execute(qSpaceFFT);
  
  // Normalise
  for(int i=0; i<(rDim[0]*rDim[1]*rDim[2]); ++i){
    qSpace[i][0] /= (rDim[0]*rDim[1]*rDim[2]);
    qSpace[i][1] /= (rDim[0]*rDim[1]*rDim[2]);
  }

  
  for(int qz=0; qz<rDim[2]; ++qz){

    int qVec[3] = {0, 0, qz};
    int qIdx = qVec[2] + rDim[2]*(qVec[1] + rDim[1]*qVec[0]);
    assert(qIdx < nspins); assert(qIdx > -1);
    int tIdx = qz + rDim[2]*timePointCounter;
    assert(tIdx < rDim[2]*nTimePoints); assert(tIdx > -1);

    tSpace[tIdx][0] = qSpace[qIdx][0];
    tSpace[tIdx][1] = qSpace[qIdx][1];
  
  }

  if(timePointCounter == (nTimePoints-1)){
    timeTransform();
    outputImage();
  }
  
  timePointCounter++;

}

void DynamicSFPhysics::timeTransform()
{
  using namespace globals;

  const int nTransforms = (nTimePoints/steps_window);
  const double normTransforms = 1.0/double(nTransforms);

  output.write("Performing %d window transforms\n",nTransforms);

  const int qzPoints    = (rDim[2]/2) + 1;
  const int omegaPoints = (steps_window/2) + 1;

  // allocate the image space
  imageSpace = static_cast<double*>(fftw_malloc(sizeof(double) * omegaPoints * qzPoints));
  fftw_complex* windowSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * 2 * steps_window * rDim[2]));

  for(int i=0; i<omegaPoints * qzPoints; ++i){
    imageSpace[i] = 0.0;
  }

  for(int i=0; i<nTransforms; ++i){ // integer division is guaranteed in the initialisation

    const int t0 = i*steps_window;
    const int tEnd = (i+1)*steps_window;

    int rank       = 1;
    int sizeN[]   = {steps_window};
    int howmany    = rDim[2];
    int inembed[] = {steps_window}; int onembed[] = {steps_window};
    int istride    = rDim[2];      int ostride    = rDim[2];
    int idist      = 1;             int odist      = 1;

    fftw_plan windowSpaceFFT = fftw_plan_many_dft(rank,sizeN,howmany,windowSpace,inembed,istride,idist,windowSpace,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
    
    // apply windowing function

    // zero the window array
    for(int t=0; t<(2*steps_window*rDim[2]); ++t){
      windowSpace[t][0] = 0.0;
      windowSpace[t][1] = 0.0;
    }

    // fill the first half of the array
    for(int t=0; t<steps_window; ++t){
      for(int qz=0; qz<rDim[2]; ++qz){
        const int tIdx = qz + rDim[2]*(t+t0);  // s_k(r ,t)
        const int refIdx = qz + rDim[2]*t0;    // s_k(r',0)
        const int wdwIdx = qz + rDim[2]*t;

        // do full complex multiplication
        const double za = tSpace[tIdx][0];
        const double zb = tSpace[tIdx][1];
        const double zc = tSpace[tIdx][0];
        const double zd = tSpace[tIdx][1];
//         const double zc = tSpace[refIdx][0];
//         const double zd = -tSpace[refIdx][1];
        windowSpace[wdwIdx][0] = (za*zc-zb*zd)*FFTWindow(t,steps_window,HANN);
        windowSpace[wdwIdx][1] = (zb*zc+za*zd)*FFTWindow(t,steps_window,HANN);
      }
    }
    
    fftw_execute(windowSpaceFFT);


    // normalise transform and apply symmetry -omega omega
    for(int t=0; t<omegaPoints;++t){
      for(int qz=0; qz<qzPoints; ++qz){
        const int tIdx = qz + rDim[2]*t;
        const int tIdxMinus = qz + rDim[2]*( (steps_window) - t);
//         assert( tIdx >= 0 );
//         assert( tIdx < (nTimePoints*rDim[2]) );
//         assert( tIdxMinus >= 0 );
//         assert( tIdxMinus < (nTimePoints*rDim[2]) );

        if(t==0){
          windowSpace[tIdx][0] = windowSpace[tIdx][0]/sqrt(double(nspins)*double(steps_window));
          windowSpace[tIdx][1] = windowSpace[tIdx][1]/sqrt(double(nspins)*double(steps_window));
        }else{
          windowSpace[tIdx][0] = 0.5*(windowSpace[tIdx][0] + windowSpace[tIdxMinus][0])/sqrt(double(nspins)*double(steps_window));
          windowSpace[tIdx][1] = 0.5*(windowSpace[tIdx][1] + windowSpace[tIdxMinus][1])/sqrt(double(nspins)*double(steps_window));
  //         windowSpace[tIdx][0] = windowSpace[tIdx][0]/sqrt(double(nspins)*double(steps_window));
  //         windowSpace[tIdx][1] = windowSpace[tIdx][1]/sqrt(double(nspins)*double(steps_window));
        }

        // assign pixels to image
        int imageIdx = qz+qzPoints*t;
        assert( imageIdx >= 0 );
        assert( imageIdx < (omegaPoints * qzPoints) );
        imageSpace[imageIdx] = imageSpace[imageIdx] + (windowSpace[tIdx][0]*windowSpace[tIdx][0]+windowSpace[tIdx][1]*windowSpace[tIdx][1])*normTransforms;
      }
    }

    fftw_destroy_plan(windowSpaceFFT);
  }
  fftw_free(windowSpace);
}

void DynamicSFPhysics::outputImage()
{
  using namespace globals;
  std::ofstream DSFFile;

  std::string filename = "_dsf.dat";
  filename = seedname+filename;
  DSFFile.open(filename.c_str());
  const int qzPoints = (rDim[2]/2)+1;
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
    case HANN:
      return 0.5*(1.0 - cos((2.0*M_PI*n)/double(nTotal-1)));
    default:
      // default to Hann window
      return 0.5*(1.0 - cos((2.0*M_PI*n)/double(nTotal-1)));
      break;
  }
}
