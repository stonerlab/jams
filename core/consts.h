// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CONSTS_H
#define JAMS_CORE_CONSTS_H

#include <complex>

const double kZero                  = 0.0;
const double kOne                   = 1.0;
const double kPi                    = 3.14159265358979323846264338327950288;
const double kTwoPi                 = 2.0*kPi;
const double kBohrMagneton          = 9.27400915E-24;
const double kGyromagneticRatio     = 1.760859770e11;
const double kBoltzmann             = 1.3806504e-23;
const double kVacuumPermeadbility   = 4*kPi*1E-7;
const double kNanometer             = 1E-9;
const double kHBar                  = 1.05457173e-34;
const double kTHz                   = 1E12;

const std::complex<double> kCmplxZero(0.0, 0.0);
const std::complex<double> kImagOne(0.0, 1.0);
const std::complex<double> kImagTwoPi(0.0, kTwoPi);

#endif  // JAMS_CORE_CONSTS_H
