// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CONSTS_H
#define JAMS_CORE_CONSTS_H

#include <complex>

const double kEps                   = 1e-5; // small number for comparing floats
const double kZero                  = 0.0;
const double kOne                   = 1.0;
const double kTwo                   = 2.0;
const double kThree                 = 3.0;
const double kSqrtTwo               = 1.41421356237309504880168872420969808;
const double kSqrtTwo_Pi            = 0.797884560802865355879892119868763737;
const double kSqrtOne_Two           = 0.707106781186547524400844362104849039;
const double kOne_SqrtTwoPi         = 0.398942280401432677939946059934381868;
const double kPi                    = 3.14159265358979323846264338327950288;
const double kTwoPi                 = 2.0*kPi;
const double kElectronGFactor       = 2.00231930436182; //         || NIST 2014 CODATA
const double kBohrMagneton          = 9.274009994E-24;  // J T^-1  || NIST 2014 CODATA
const double kGyromagneticRatio     = 1.760859644e11;   // s^-1 T  || NIST 2014 CODATA
const double kBoltzmann             = 1.38064852e-23;   // J K^-1  || NIST 2014 CODATA
const double kVacuumPermeadbility   = 4*kPi*1E-7;
const double kNanometer             = 1E-9;
const double kHBar                  = 1.054571800e-34;  // J s     || NIST 2014 CODATA
const double kTHz                   = 1E12;

const std::complex<double> kCmplxZero(0.0, 0.0);
const std::complex<double> kImagOne(0.0, 1.0);
const std::complex<double> kImagTwoPi(0.0, kTwoPi);

#endif  // JAMS_CORE_CONSTS_H
