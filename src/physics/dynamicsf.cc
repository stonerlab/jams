#include "globals.h"
#include "dynamicsf.h"

#include <fftw3.h>
#include <string>
#include <sstream>
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

	if(config.exists("physics.typewise")){
		config.lookupValue("physics.typewise",typeToggle);
		if(typeToggle == true){
			output.write("  * Typewise transforms enabled\n");
			
		}
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

    nBZPoints += 1; // include last symmetry point
	

	// calculate Brillouin zone vectors points
	int vec[3] = {0,0,0};
	int counter=0;

    // go through once to count total number of points with symmetry
	for(int i=0; i<(nSymPoints-1); ++i){
		for(int j=0; j<3; ++j){
			vec[j] = SymPoints(i+1,j)-SymPoints(i,j);
			if(vec[j] != 0){
				vec[j] = vec[j] / abs(vec[j]);
			}
		}
		for(int n=0; n<BZPointCount(i); ++n){
			int bzvec[3];
			for(int j=0; j<3; ++j){
				bzvec[j] = abs(SymPoints(i,j)+n*vec[j]);
			}
	        std::sort(bzvec,bzvec+3);
	        do {
				counter++;
	        } while (next_point_symmetry(bzvec));
		}
	}
	{
		int bzvec[3];
		for(int j=0; j<3; ++j){
			bzvec[j] = abs(SymPoints(nSymPoints-1,j));
		}
        std::sort(bzvec,bzvec+3);
        do {
			counter++;
        } while (next_point_symmetry(bzvec));		
	}

    // go through again and create arrays
	BZIndex.resize(nBZPoints+1);
	BZPoints.resize(counter+1,3); // +1 to add on final point
    BZLengths.resize(nBZPoints);
	BZDegeneracy.resize(nBZPoints);
	for(int i=0; i<nBZPoints; ++i){BZDegeneracy(i)=0;}
	
	int irreducibleCounter=0;
    counter=0;
	for(int i=0; i<(nSymPoints-1); ++i){
		for(int j=0; j<3; ++j){
			vec[j] = SymPoints(i+1,j)-SymPoints(i,j);
			if(vec[j] != 0){
				vec[j] = vec[j] / abs(vec[j]);
			}
		}
		for(int n=0; n<BZPointCount(i); ++n){
            BZLengths(irreducibleCounter) = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
			BZIndex(irreducibleCounter) = counter;
			int bzvec[3];
			int pbcvec[3];
			for(int j=0; j<3; ++j){
				bzvec[j] = abs(SymPoints(i,j)+n*vec[j]);
			}
	        std::sort(bzvec,bzvec+3);
			output.write("BZ Point: %8d %8d [ %4d %4d %4d ]\n", irreducibleCounter, counter, bzvec[0], bzvec[1], bzvec[2]);
	        do {
				// apply periodic boundaries here
				// FFTW stores -q in reverse order at the end of the array.
				for(int j=0; j<3; ++j){
				  pbcvec[j] = ((qDim[j])+bzvec[j])%(qDim[j]);
				  BZPoints(counter,j) = pbcvec[j];
			  	}
			  //std::cout<<bzvec[0]<<"\t"<<bzvec[1]<<"\t"<<bzvec[2]<<"\t"<<pbcvec[0]<<"\t"<<pbcvec[1]<<"\t"<<pbcvec[2]<<std::endl;
				
				counter++;
				BZDegeneracy(irreducibleCounter)++;
	        } while (next_point_symmetry(bzvec));
			irreducibleCounter++;
		}
		BZIndex(irreducibleCounter) = counter;
	}

    // include last point on curve
	{
		for(int j=0; j<3; ++j){
			vec[j] = SymPoints(nSymPoints-1,j);
			if(vec[j] != 0){
				vec[j] = vec[j] / abs(vec[j]);
			}
		}
		BZLengths(irreducibleCounter) = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		BZIndex(irreducibleCounter) = counter;
		int bzvec[3];
		int pbcvec[3];
		for(int j=0; j<3; ++j){
			bzvec[j] = abs(SymPoints(nSymPoints-1,j));
		}
		std::sort(bzvec,bzvec+3);
		output.write("BZ Point: %8d %8d [ %4d %4d %4d ]\n", irreducibleCounter, counter, bzvec[0], bzvec[1], bzvec[2]);
		do {
			// apply periodic boundaries here
			// FFTW stores -q in reverse order at the end of the array.
			for(int j=0; j<3; ++j){
				pbcvec[j] = ((qDim[j])+bzvec[j])%(qDim[j]);
				BZPoints(counter,j) = pbcvec[j];
			}
			BZDegeneracy(irreducibleCounter)++;
			counter++;
		} while (next_point_symmetry(bzvec));
		irreducibleCounter++;
	}
	BZIndex(irreducibleCounter) = counter;
	
	
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
	
	if(typeToggle == true){
		output.write("  * qSpace allocating %f MB\n", (sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2]*lattice.numTypes())/(1024.0*1024.0));
		qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2]*lattice.numTypes()));
		
		if(qSpace == NULL){
			jams_error("Failed to allocate qSpace FFT array");
		}
	
		const int qTotal = qDim[0]*qDim[1]*qDim[2];
		for(int i=0; i<lattice.numTypes();++i){
			qSpaceFFT.push_back(fftw_plan_dft_3d(qDim[0],qDim[1],qDim[2],qSpace+i*qTotal,qSpace+i*qTotal,FFTW_FORWARD,FFTW_MEASURE));
		}
	} else {
		output.write("  * qSpace allocating %f MB\n", (sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2])/(1024.0*1024.0));
		qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2]));
		
		if(qSpace == NULL){
			jams_error("Failed to allocate qSpace FFT array");
		}
	
		qSpaceFFT.push_back(fftw_plan_dft_3d(qDim[0],qDim[1],qDim[2],qSpace,qSpace,FFTW_FORWARD,FFTW_MEASURE));
	}


	
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
	if(typeToggle == true){
		output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*lattice.numTypes()*nTimePoints*nBZPoints/(1024.0*1024.0)));

		tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*lattice.numTypes()*nTimePoints*nBZPoints));
		for(int i=0; i<nTimePoints*nBZPoints*lattice.numTypes(); ++i){
			tSpace[i][0] = 0.0;
			tSpace[i][1] = 0.0;
		}
		
	} else {
		output.write("  * tSpace allocating %f MB\n", (sizeof(fftw_complex)*nTimePoints*nBZPoints/(1024.0*1024.0)));

		tSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nTimePoints*nBZPoints));
		for(int i=0; i<nTimePoints*nBZPoints; ++i){
			tSpace[i][0] = 0.0;
			tSpace[i][1] = 0.0;
		}
	}

	if(tSpace == NULL){
		jams_error("Failed to allocate tSpace FFT array");
	}

	initialised = true;
}

DynamicSFPhysics::~DynamicSFPhysics()
{
	using namespace globals;
	if(initialised == true){
		if(typeToggle==true){
			for(int i=0;i<lattice.numTypes();++i){
				fftw_destroy_plan(qSpaceFFT[i]);
			}
		}else{
			fftw_destroy_plan(qSpaceFFT[0]);
		}

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

	const int qTotal = qDim[0]*qDim[1]*qDim[2];
	
	if(typeToggle == true){
		// Apply cofactors to transform spin components
			if(componentImag == -1){
				for(int i=0; i<nspins; ++i){
					const int type = lattice.getType(i);
					const int idx = spinToKspaceMap[i];
					qSpace[idx+qTotal*type][0] = coFactors(type,componentReal)*s(i,componentReal);
					qSpace[idx+qTotal*type][1] = 0.0;
				}
			} else {
				for(int i=0; i<nspins; ++i){
					const int type = lattice.getType(i);
					const int idx = spinToKspaceMap[i];
					qSpace[idx+qTotal*type][0] = coFactors(type,componentReal)*s(i,componentReal);
					qSpace[idx+qTotal*type][1] = coFactors(type,componentImag)*s(i,componentImag);
				}
			}

		for(int i=0; i<lattice.numTypes();++i){
			
			fftw_execute(qSpaceFFT[i]);
		}
			

	// Normalise FFT by Nx*Ny*Nz
			for(int i=0; i<lattice.numTypes();++i){
				for(int j=0; j<qTotal; ++j){
					qSpace[i*qTotal+j][0] /= (qTotal);
					qSpace[i*qTotal+j][1] /= (qTotal);
				}
			}
		
	} else {
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

		fftw_execute(qSpaceFFT[0]);

	// Normalise FFT by Nx*Ny*Nz
		for(int i=0; i<(qDim[0]*qDim[1]*qDim[2]); ++i){
			qSpace[i][0] /= (qDim[0]*qDim[1]*qDim[2]);
			qSpace[i][1] /= (qDim[0]*qDim[1]*qDim[2]);
		}
	}


	// note periodic boundary conditions we applied in the initialisation
	if(typeToggle == true){
		for(int n=0; n<nBZPoints; ++n){
			for(int q=BZIndex(n); q<BZIndex(n+1); ++q){
				for(int i=0; i<lattice.numTypes();++i){
					const int qVec[3] = {BZPoints(q,0), BZPoints(q,1), BZPoints(q,2)};
					const int qIdx = qVec[2] + qDim[2]*(qVec[1] + qDim[1]*qVec[0]);
					const int tIdx = n + nBZPoints*timePointCounter;
		
					assert(qIdx < nspins); 
					assert(qIdx > -1);
					assert(tIdx < nBZPoints*nTimePoints); 
					assert(tIdx > -1);

					tSpace[tIdx+i*nTimePoints*nBZPoints][0] = qSpace[qIdx+i*qTotal][0];
					tSpace[tIdx+i*nTimePoints*nBZPoints][1] = qSpace[qIdx+i*qTotal][1];
				}
			}
		}
	}else{
		for(int n=0; n<nBZPoints; ++n){
			for(int q=BZIndex(n); q<BZIndex(n+1); ++q){
				const int qVec[3] = {BZPoints(q,0), BZPoints(q,1), BZPoints(q,2)};
				const int qIdx = qVec[2] + qDim[2]*(qVec[1] + qDim[1]*qVec[0]);
				const int tIdx = n + nBZPoints*timePointCounter;
		
				assert(qIdx < nspins); 
				assert(qIdx > -1);
				assert(tIdx < nBZPoints*nTimePoints); 
				assert(tIdx > -1);

				tSpace[tIdx][0] = qSpace[qIdx][0];
				tSpace[tIdx][1] = qSpace[qIdx][1];
			}
		}		
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
	
	if(typeToggle==true){
		imageSpace = static_cast<double*>(fftw_malloc(sizeof(double) * omegaPoints * nBZPoints * lattice.numTypes()));
		for(int i=0; i<omegaPoints * nBZPoints * lattice.numTypes(); ++i){
			imageSpace[i] = 0.0;
		}
	}else{
		imageSpace = static_cast<double*>(fftw_malloc(sizeof(double) * omegaPoints * nBZPoints));
		for(int i=0; i<omegaPoints * nBZPoints; ++i){
			imageSpace[i] = 0.0;
		}	
	}
	

		
	for(int n=0; n<nTransforms; ++n){ // integer division is guaranteed in the initialisation

		const int t0 = n*steps_window;
		const int tEnd = (n+1)*steps_window;

		int rank       = 1;
		int sizeN[]   = {steps_window};
		int howmany    = nBZPoints;
		int inembed[] = {steps_window}; int onembed[] = {steps_window};
		int istride    = nBZPoints;      int ostride    = nBZPoints;
		int idist      = 1;             int odist      = 1;
		fftw_complex* startPtr = (tSpace+n*steps_window*nBZPoints); // pointer arithmatic

		fftw_plan tSpaceFFT = fftw_plan_many_dft(rank,sizeN,howmany,startPtr,inembed,istride,idist,startPtr,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
    
    // apply windowing function
		if(typeToggle==true){
			for(int i=0; i<lattice.numTypes();++i){

				for(unsigned int t=0; t<steps_window; ++t){
					for(int q=0; q<nBZPoints; ++q){
						const int tIdx = q + nBZPoints*(t+t0);
						tSpace[tIdx][0] = tSpace[tIdx+i*nBZPoints*nTimePoints][0]*FFTWindow(t,steps_window,HAMMING);
						tSpace[tIdx][1] = tSpace[tIdx+i*nBZPoints*nTimePoints][1]*FFTWindow(t,steps_window,HAMMING);
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

						tSpace[tIdx][0] = tSpace[tIdx][0]/sqrt(double(nspins)*double(steps_window));
						tSpace[tIdx][1] = tSpace[tIdx][1]/sqrt(double(nspins)*double(steps_window));

				//tSpace[tIdx][0] = 0.5*(tSpace[tIdx][0] + tSpace[tIdxMinus][0])/sqrt(double(nspins)*double(steps_window));
				//tSpace[tIdx][1] = 0.5*(tSpace[tIdx][1] + tSpace[tIdxMinus][1])/sqrt(double(nspins)*double(steps_window));

        // zero -omega to avoid accidental use
						tSpace[tIdxMinus][0] = 0.0; tSpace[tIdxMinus][1] = 0.0;

        // assign pixels to image
						int imageIdx = q+nBZPoints*t;
						assert( imageIdx >= 0 );
						assert( imageIdx < (omegaPoints * nBZPoints) );
						imageSpace[imageIdx+i*omegaPoints*nBZPoints] = imageSpace[imageIdx+i*omegaPoints*nBZPoints] + (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1])*normTransforms;
					}
				}
			}
		
		}else{

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

					tSpace[tIdx][0] = tSpace[tIdx][0]/sqrt(double(nspins)*double(steps_window));
					tSpace[tIdx][1] = tSpace[tIdx][1]/sqrt(double(nspins)*double(steps_window));

					//tSpace[tIdx][0] = 0.5*(tSpace[tIdx][0] + tSpace[tIdxMinus][0])/sqrt(double(nspins)*double(steps_window));
					//tSpace[tIdx][1] = 0.5*(tSpace[tIdx][1] + tSpace[tIdxMinus][1])/sqrt(double(nspins)*double(steps_window));

	        // zero -omega to avoid accidental use
					tSpace[tIdxMinus][0] = 0.0; tSpace[tIdxMinus][1] = 0.0;

	        // assign pixels to image
					int imageIdx = q+nBZPoints*t;
					assert( imageIdx >= 0 );
					assert( imageIdx < (omegaPoints * nBZPoints) );
					imageSpace[imageIdx] = imageSpace[imageIdx] + (tSpace[tIdx][0]*tSpace[tIdx][0] + tSpace[tIdx][1]*tSpace[tIdx][1])*normTransforms;
				}
			}
		}	
			
		startPtr = NULL;
    
		fftw_destroy_plan(tSpaceFFT);
	}
}


void DynamicSFPhysics::outputImage()
{
	using namespace globals;
	const int omegaPoints = (steps_window/2) + 1;
	
	if(typeToggle == true){
	for(int i=0; i<lattice.numTypes();++i){
		std::ofstream DSFFile;

	    std::ostringstream ss;
		std::string filename;
		
		ss << seedname << "_dsf_" << i << ".dat";
		filename = ss.str();
		DSFFile.open(filename.c_str());
		float lengthTotal=0.0;
	
		std::vector<double> dos((steps_window/2)+1,0.0);
	
		for(int n=0; n<nBZPoints; ++n){
			const int q = BZIndex(n);
			for(unsigned int omega=0; omega<((steps_window/2)+1); ++omega){
				int tIdx = n + nBZPoints*omega;
				DSFFile << lengthTotal << "\t" << BZPoints(q,0) << "\t" <<BZPoints(q,1) <<"\t"<<BZPoints(q,2);
				DSFFile << "\t" << omega*freqIntervalSize <<"\t" << imageSpace[tIdx+i*omegaPoints*nBZPoints]<<"\t"<<static_cast<double>(BZDegeneracy(n))<<"\n";
				dos[omega] += imageSpace[tIdx+i*omegaPoints*nBZPoints];
			}
			DSFFile << std::endl;
			lengthTotal += BZLengths(n);
		}
  
		DSFFile.close();
	
		std::ofstream DOSFile;
		
		ss.str("");
		ss << seedname << "_dos_" << i << ".dat";
		filename = ss.str();
		DSFFile.open(filename.c_str());

		DOSFile.open(filename.c_str());
		for(unsigned int omega=0; omega<((steps_window/2)+1); ++omega){
			DOSFile << omega*freqIntervalSize <<"\t" << dos[omega] <<"\n";
		}
	
		DOSFile.close();
	}
	}else{
		std::ofstream DSFFile;

	    std::ostringstream ss;
		std::string filename;
		
		ss << seedname << "_dsf.dat";
		filename = ss.str();
		DSFFile.open(filename.c_str());
		float lengthTotal=0.0;
	
		std::vector<double> dos((steps_window/2)+1,0.0);
	
		for(int n=0; n<nBZPoints; ++n){
			const int q = BZIndex(n);
			for(unsigned int omega=0; omega<((steps_window/2)+1); ++omega){
				int tIdx = n + nBZPoints*omega;
				DSFFile << lengthTotal << "\t" << BZPoints(q,0) << "\t" <<BZPoints(q,1) <<"\t"<<BZPoints(q,2);
				DSFFile << "\t" << omega*freqIntervalSize <<"\t" << imageSpace[tIdx]<<"\t"<<static_cast<double>(BZDegeneracy(n))<<"\n";
				dos[omega] += imageSpace[tIdx];
			}
			DSFFile << std::endl;
			lengthTotal += BZLengths(n);
		}
  
		DSFFile.close();
	
		std::ofstream DOSFile;
		
		ss.str("");
		ss << seedname << "_dos.dat";
		filename = ss.str();
		DSFFile.open(filename.c_str());

		DOSFile.open(filename.c_str());
		for(unsigned int omega=0; omega<((steps_window/2)+1); ++omega){
			DOSFile << omega*freqIntervalSize <<"\t" << dos[omega] <<"\n";
		}
	
		DOSFile.close();
	}
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
