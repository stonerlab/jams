#include "globals.h"
#include "dynamicsf.h"

#include <fftw3.h>
#include <string>
#include <map>
#include "maths.h"

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

	lattice.getKspaceDimensions(qDim[0],qDim[1],qDim[2]);
	output.write("  * Kspace Size [%d,%d,%d]\n",qDim[0],qDim[1],qDim[2]);
	
	
	// read irreducible Brillouin zone
	const int nSymPoints = phys["brillouinzone"].getLength();
	
	Array2D<int> SymPoints(nSymPoints,3);
	Array<int> BZPointCount(nSymPoints-1);
	
	
	for(int i=0; i<nSymPoints; ++i){
		for(int j=0; j<3; ++j){
			SymPoints(i,j) = phys["brillouinzone"][i][j];
		}
	}
	
	// count number of Brillouin zone vector points we need
	nBZPoints = 0;
	for(int i=0; i<(nSymPoints-1); ++i){
		int max=0;
		for(int j=0; j<3; ++j){
			int x = abs(SymPoints(i+1,j) - SymPoints(i,j));
			if (x > max){
				max = x;
			}
		}
		BZPointCount(i) = max;
		nBZPoints += max;
	}
	
	// calculate Brillouin zone vectors points
	BZPoints.resize(nBZPoints,3);
	int counter=0;
	for(int i=0; i<(nSymPoints-1); ++i){
		int vec[3];
		for(int j=0; j<3; ++j){
			vec[j] = SymPoints(i+1,j)-SymPoints(i,j);
			if(vec[j] != 0){
				vec[j] = vec[j] / abs(vec[j]);
			}
		}
		for(int n=0; n<BZPointCount(i); ++n){
			for(int j=0; j<3; ++j){
				BZPoints(counter,j) = SymPoints(i,j)+n*vec[j];
			}
			output.write("BZ Point: %8d [ %4d %4d %4d ]\n", counter, BZPoints(counter,0), BZPoints(counter,1), BZPoints(counter,2));
			counter++;
		}
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
    steps_window = nint(t_window/(t_out));
    output.write("  * Window samples: %d\n",steps_window);
	if(nTimePoints%(steps_window) != 0){
		jams_error("Window time must be an integer multiple of the run time");
	}

	freqIntervalSize = (M_PI)/(sampletime*steps_window);
	output.write("  * Sample frequency: %f [GHz]\n",freqIntervalSize/1E9);


	output.write("  * Initialising FFTW variables...\n");

// --------------------------------------------------------------------------------------------------------------------
// Real space to reciprocal (q) space transform
// --------------------------------------------------------------------------------------------------------------------
	output.write("  * qSpace allocating %f MB\n", (sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2])/(1024.0*1024.0));
	qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2]));
	if(qSpace == NULL){
		jams_error("Failed to allocate qSpace FFT array");
	}
	qSpaceFFT = fftw_plan_dft_3d(qDim[0],qDim[1],qDim[2],qSpace,qSpace,FFTW_FORWARD,FFTW_MEASURE);

	
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


        int idx = (x*qDim[1]+y)*qDim[2]+z;
        spinToKspaceMap[i] = idx;
    }

// --------------------------------------------------------------------------------------------------------------------
// Time to frequency space transform
// --------------------------------------------------------------------------------------------------------------------
	output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*nBZPoints/(1024.0*1024.0)));

	tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*nBZPoints));

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

	// Apply cofactors to transform spin components
	if(componentImag == -1){
		for(int i=0; i<nspins; ++i){
			const int type = lattice.getType(i);
			const int idx = spinToKspaceMap[i];
			qSpace[idx][0] = coFactors(type,componentReal)*s(i,componentReal);
			qSpace[idx][1] = 0.0;
		}
	} else {
		for(int i=0; i<nspins; ++i){
			const int type = lattice.getType(i);
			const int idx = spinToKspaceMap[i];
			qSpace[idx][0] = coFactors(type,componentReal)*s(i,componentReal);
			qSpace[idx][1] = coFactors(type,componentImag)*s(i,componentImag);
		}
	}

	fftw_execute(qSpaceFFT);

	// Normalise FFT by Nx*Ny*Nz
	for(int i=0; i<(qDim[0]*qDim[1]*qDim[2]); ++i){
		qSpace[i][0] /= (qDim[0]*qDim[1]*qDim[2]);
		qSpace[i][1] /= (qDim[0]*qDim[1]*qDim[2]);
	}

	for(int q=0; q<nBZPoints; ++q){

		const int qVec[3] = {BZPoints(q,0), BZPoints(q,1), BZPoints(q,2)};
		const int qIdx = qVec[2] + qDim[2]*(qVec[1] + qDim[1]*qVec[0]);
		const int tIdx = q + nBZPoints*timePointCounter;
		
		assert(qIdx < nspins); 
		assert(qIdx > -1);
		assert(tIdx < nBZPoints*nTimePoints); 
		assert(tIdx > -1);

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

	const int omegaPoints = (steps_window/2) + 1;

  // allocate the image space
	imageSpace = static_cast<double*>(fftw_malloc(sizeof(double) * omegaPoints * nBZPoints));
	for(int i=0; i<omegaPoints * nBZPoints; ++i){
		imageSpace[i] = 0.0;
	}

	for(int i=0; i<nTransforms; ++i){ // integer division is guaranteed in the initialisation

		const int t0 = i*steps_window;
		const int tEnd = (i+1)*steps_window;

		int rank       = 1;
		int sizeN[]   = {steps_window};
		int howmany    = nBZPoints;
		int inembed[] = {steps_window}; int onembed[] = {steps_window};
		int istride    = nBZPoints;      int ostride    = nBZPoints;
		int idist      = 1;             int odist      = 1;
		fftw_complex* startPtr = (tSpace+i*steps_window*nBZPoints); // pointer arithmatic

		fftw_plan tSpaceFFT = fftw_plan_many_dft(rank,sizeN,howmany,startPtr,inembed,istride,idist,startPtr,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
    
    // apply windowing function

		for(unsigned int t=0; t<steps_window; ++t){
			for(int q=0; q<nBZPoints; ++q){
				const int tIdx = q + nBZPoints*(t+t0);
				tSpace[tIdx][0] = tSpace[tIdx][0]*FFTWindow(t,steps_window,HAMMING);
				tSpace[tIdx][1] = tSpace[tIdx][1]*FFTWindow(t,steps_window,HAMMING);
			}
		}
    
		fftw_execute(tSpaceFFT);


		// normalise transform and apply symmetry -omega omega
		for(int t=0; t<omegaPoints;++t){
			for(int q=0; q<nBZPoints; ++q){
				const int tIdx = q + nBZPoints*(t0+t);
				const int tIdxMinus = q + nBZPoints*( (tEnd-1) - t);
				assert( tIdx >= 0 );
				assert( tIdx < (nTimePoints*nBZPoints) );
				assert( tIdxMinus >= 0 );
				assert( tIdxMinus < (nTimePoints*nBZPoints) );

				tSpace[tIdx][0] = 0.5*(tSpace[tIdx][0] + tSpace[tIdxMinus][0])/sqrt(double(nspins)*double(steps_window));
				tSpace[tIdx][1] = 0.5*(tSpace[tIdx][1] + tSpace[tIdxMinus][1])/sqrt(double(nspins)*double(steps_window));

        // zero -omega to avoid accidental use
				tSpace[tIdxMinus][0] = 0.0; tSpace[tIdxMinus][1] = 0.0;

        // assign pixels to image
				int imageIdx = q+nBZPoints*t;
				assert( imageIdx >= 0 );
				assert( imageIdx < (omegaPoints * nBZPoints) );
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
	for(int q=0; q<nBZPoints; ++q){
		for(unsigned int omega=0; omega<((steps_window/2)+1); ++omega){
			int tIdx = q + nBZPoints*omega;
			DSFFile << BZPoints(q,0) << "\t" <<BZPoints(q,1) <<"\t"<<BZPoints(q,2) << "\t" << omega*freqIntervalSize <<"\t" << imageSpace[tIdx] <<"\n";
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
