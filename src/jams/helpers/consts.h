// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CONSTS_H
#define JAMS_CORE_CONSTS_H

#include <complex>

constexpr double kZero                  = 0.0;
constexpr double kOne                   = 1.0;
constexpr double kTwo                   = 2.0;
constexpr double kThree                 = 3.0;
constexpr double kSqrtTwo               = 1.41421356237309504880168872420969808;
constexpr double kSqrtTwo_Pi            = 0.797884560802865355879892119868763737;
constexpr double kSqrtOne_Two           = 0.707106781186547524400844362104849039;
constexpr double kOne_SqrtTwoPi         = 0.398942280401432677939946059934381868;
constexpr double kPi                    = 3.14159265358979323846264338327950288;
constexpr double kTwoPi                 = 2.0*kPi;
constexpr double kElectronGFactor       = 2.0023193043625; //         || NIST 2018 CODATA
constexpr double kNeutronGFactor        = 3.82608545; //               || NIST (https://physics.nist.gov)
constexpr double kElementaryCharge      = 1.602176634e-19;  // C       || NIST (https://physics.nist.gov)
constexpr double kElectronMass          = 9.1093837015e-31; // kg      || NIST (https://physics.nist.gov)
constexpr double kSpeedOfLight          = 299792458.0;      // m s^-1  || NIST (https://physics.nist.gov)
constexpr double kNanometer             = 1E-9;
constexpr double kTHz                   = 1E12;

// IU - means internal units
// time -> picoseconds (ps)
// fields -> Tesla (T)
// energy -> millielectron volts (meV)

constexpr double kJoule2meV = 6.24150907e21; // 1 Joule in meV
constexpr double kmRyd2meV  = 13.605693123; // 1 mRyd in meV

constexpr double kHBarIU              = 0.6582119569;  // meV ps
constexpr double kBohrMagnetonIU      = 0.0578838181;  // meV T^-1
constexpr double kGyromagneticRatioIU = kElectronGFactor * kBohrMagnetonIU / kHBarIU;  // rad ps^-1 T^-1
constexpr double kBoltzmannIU         = 0.0861733326;  // meV K^-1
constexpr double kVacuumPermeabilityIU   = 4*kPi*1E-7 / kJoule2meV; // Original is H m^-1 = J A^-2 m^-1 which we change to

constexpr double kMeterToAngstroms      = 1e10;

constexpr double kBytesToMegaBytes      = 1048576.0;

constexpr std::complex<double> kCmplxZero = {0.0, 0.0};
constexpr std::complex<double> kImagOne = {0.0, 1.0};
constexpr std::complex<double> kImagTwoPi = {0.0, kTwoPi};

#endif  // JAMS_CORE_CONSTS_H
