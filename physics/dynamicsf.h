// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_DYNAMICSF_H
#define JAMS_PHYSICS_DYNAMICSF_H

#include <fftw3.h>

#include <vector>

#include "core/physics.h"

#include "jblib/containers/array.h"

enum FFTWindowType {GAUSSIAN, HAMMING};

class DynamicSFPhysics : public Physics {
 public:
  DynamicSFPhysics(const libconfig::Setting &settings);
  ~DynamicSFPhysics();

  void update(const int &iterations, const double &time, const double &dt);

 private:
  bool              initialized;
  bool        typeToggle;
  int               timePointCounter;
  int               nTimePoints;
  std::vector<int>  qDim;
  fftw_complex*     qSpace;
  fftw_complex*     tSpace;
  double *          imageSpace;
  std::vector<fftw_plan>  qSpaceFFT;
  int               componentReal;
  int               componentImag;
  jblib::Array<double, 2>   coFactors;
  double            freqIntervalSize;
  double            t_window;
  unsigned long     steps_window;
  std::vector<int>  spinToKspaceMap;
  int               nBZPoints;
  jblib::Array<int, 1>      BZIndex;
  jblib::Array<int, 2>      BZPoints;
  jblib::Array<int, 1>      BZDegeneracy;
  jblib::Array<float, 1>      BZLengths;

  double FFTWindow(const int n, const int nTotal, const FFTWindowType type);
  void   timeTransform();
  void   outputImage();
};

#endif /* JAMS_PHYSICS_DYNAMICSF_H */
