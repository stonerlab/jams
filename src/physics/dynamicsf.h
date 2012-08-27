#ifndef __DYNAMICSF_H__
#define __DYNAMICSF_H__

#include "physics.h"

#include <vector>
#include <fftw3.h>

enum FFTWindowType {GAUSSIAN,HAMMING};

class DynamicSFPhysics : public Physics {
public:
	DynamicSFPhysics()
		: initialised(false),
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
            BZPoints(),
            BZLengths() {}
    
		~DynamicSFPhysics();
    
		void init(libconfig::Setting &phys);
		void run(double realtime, const double dt);
		virtual void monitor(double realtime, const double dt);

	private:
		bool              initialised;
		int               timePointCounter;
		int               nTimePoints;
		std::vector<int>  qDim;
		fftw_complex*     qSpace;
		fftw_complex*     tSpace;
		double *          imageSpace;
		fftw_plan         qSpaceFFT;
		int               componentReal;
		int               componentImag;
		Array2D<double>   coFactors;
		double            freqIntervalSize;
		double            t_window;
		unsigned long     steps_window;
		std::vector<int>  spinToKspaceMap;
        int               nBZPoints;
        Array2D<int>      BZPoints;
        Array<float>      BZLengths;

		double FFTWindow(const int n, const int nTotal, const FFTWindowType type); 
		void   timeTransform();
		void   outputImage();

	};

#endif /* __DYNAMICSF_H__ */
