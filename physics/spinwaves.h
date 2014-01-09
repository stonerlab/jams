// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHSYICS_SPINWAVES_H
#define JAMS_PHSYICS_SPINWAVES_H

#include <fftw3.h>
#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "core/physics.h"

#include "jblib/containers/array.h"

class SpinwavesPhysics : public Physics {
 public:
  SpinwavesPhysics()
  : qDim(3, 0),
  qSpace(),
  qSpaceFFT(0),
  componentReal(0),
  componentImag(0),
  coFactors(0, 0),
  spinToKspaceMap(0),
  nBZPoints(0),
  BZIndex(0),
  BZPoints(0, 0),
  BZDegeneracy(0),
  BZLengths(0),
  BZData(0),
  SPWFile(),
  ModeFile(),
  SPDFile(),
  typeOverride(),
  initialised(false),
  squareToggle(false),
  pumpTime(0.0),
  pumpStartTime(0.0),
  pumpTemp(0.0),
  pumpFluence(0.0),
  electronTemp(0.0),
  phononTemp(0.0),
  reversingField(3, 0.0),
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
  std::vector<int>        qDim;
  fftw_complex*           qSpace;
  std::vector<fftw_plan>  qSpaceFFT;
  int                     componentReal;
  int                     componentImag;
  jblib::Array<double, 2> coFactors;
  std::vector<int>        spinToKspaceMap;
  int                     nBZPoints;
  jblib::Array<int, 1>    BZIndex;
  jblib::Array<int, 2>    BZPoints;
  jblib::Array<int, 1>    BZDegeneracy;
  jblib::Array<float, 1>  BZLengths;
  jblib::Array<float, 1>  BZData;

  std::ofstream   SPWFile;
  std::ofstream   ModeFile;
  std::ofstream   SPDFile;
  std::vector<int> typeOverride;
  bool initialised;
  bool squareToggle;

    // calculation of pump power which is linear with input approx
    // electron temperature
  double pumpPower(double &pF) { return (1.152E20*pF); }

  double pumpTime;
  double pumpStartTime;
  double pumpTemp;
  double pumpFluence;
  double electronTemp;
  double phononTemp;
  std::vector<double> reversingField;

  double Ce;  // electron specific heat
  double Cl;  // phonon specific heat
  double G;   // electron coupling constant

  std::ofstream TTMFile;
};

#endif  // JAMS_PHSYICS_SPINWAVES_H
