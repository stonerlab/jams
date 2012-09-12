#ifndef __SPINWAVES_H__
#define __SPINWAVES_H__

#include <fstream>
#include <vector>
#include <libconfig.h++>
#include <fftw3.h>
#include "physics.h"

class SpinwavesPhysics : public Physics {
  public:
    SpinwavesPhysics()
      : qDim(3,0),
		qSpace(),
		qSpaceFFT(0),
		componentReal(0),
		componentImag(0),
		coFactors(0,0),
		spinToKspaceMap(0),
		nBZPoints(0),
		BZIndex(0),
		BZPoints(0,0),
		BZDegeneracy(0),
		BZLengths(0),
		BZData(0),
        SPWFile(),
        ModeFile(),
        SPDFile(),
        typeOverride(),
        initialised(false),
        spinDump(false),        
        pumpTime(0.0),
        pumpStartTime(0.0),
        pumpTemp(0.0),
        pumpFluence(0.0),
        electronTemp(0.0),
        phononTemp(0.0),
        reversingField(3,0.0),
        Ce(7.0E02),
        Cl(3.0E06),
        G(17.0E17),
        TTMFile()
      {}

    ~SpinwavesPhysics();

    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);

  private:
	  std::vector<int>  		qDim;
	  fftw_complex*     		qSpace;
	  std::vector<fftw_plan>  	qSpaceFFT;
	  int               		componentReal;
	  int               		componentImag;
	  Array2D<double>   		coFactors;
	  std::vector<int>  		spinToKspaceMap;
      int               		nBZPoints;
	  Array<int>		  		BZIndex;
      Array2D<int>      		BZPoints;
	  Array<int>		  		BZDegeneracy;
      Array<float>      		BZLengths;
	  Array<float>				BZData;
	  
	
	  std::ofstream   SPWFile;
	  std::ofstream   ModeFile;
	  std::ofstream   SPDFile;
	  std::vector<int> typeOverride;
	  bool initialised;
	  bool spinDump;

    // calculation of pump power which is linear with input approx
    // electron temperature
	  double pumpPower(double &pF){return (1.152E20*pF);}

	  double pumpTime;
	  double pumpStartTime;
	  double pumpTemp;
	  double pumpFluence;
	  double electronTemp;
	  double phononTemp;
	  std::vector<double> reversingField;

	  double Ce; // electron specific heat
	  double Cl; // phonon specific heat
	  double G;  // electron coupling constant

	  std::ofstream TTMFile;

};
#endif /* __SPINWAVES_H__ */
