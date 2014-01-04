#ifndef JAMS_PHYSICS_DYNAMICSF_H
#define JAMS_PHYSICS_DYNAMICSF_H

#include "physics.h"

#include <vector>
#include <fftw3.h>
#include <containers/array.h>

enum FFTWindowType {GAUSSIAN,HAMMING};

class DynamicSFPhysics : public Physics {
public:
	DynamicSFPhysics()
		: initialised(false),
			typeToggle(false),
			timePointCounter(0),
			nTimePoints(0),
			qDim(3,0),
			qSpace(NULL),
			tSpace(NULL),
			imageSpace(NULL),
			qSpaceFFT(),
			componentReal(0),
			componentImag(0),
			coFactors(0,0),
			freqIntervalSize(0),
			t_window(0.0),
			steps_window(0), 
            nBZPoints(0),
			BZIndex(),	
            BZPoints(),
			BZDegeneracy(),
            BZLengths() {}
    
		~DynamicSFPhysics();
    
		void init(libconfig::Setting &phys);
		void run(double realtime, const double dt);
		virtual void monitor(double realtime, const double dt);

	private:
    bool              initialised;
    bool			  typeToggle;
    int               timePointCounter;
    int               nTimePoints;
    std::vector<int>  qDim;
    fftw_complex*     qSpace;
    fftw_complex*     tSpace;
    double *          imageSpace;
    std::vector<fftw_plan>  qSpaceFFT;
    int               componentReal;
    int               componentImag;
    jblib::Array<double,2>   coFactors;
    double            freqIntervalSize;
    double            t_window;
    unsigned long     steps_window;
    std::vector<int>  spinToKspaceMap;
    int               nBZPoints;
    jblib::Array<int,1>		  BZIndex;
    jblib::Array<int,2>      BZPoints;
    jblib::Array<int,1>		  BZDegeneracy;
    jblib::Array<float,1>      BZLengths;

		double FFTWindow(const int n, const int nTotal, const FFTWindowType type); 
		void   timeTransform();
		void   outputImage();

	};

#endif /* JAMS_PHYSICS_DYNAMICSF_H */
