// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CONSTS_H
#define JAMS_CORE_CONSTS_H

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

const double pi = 3.14159265358979323846;
const double mu_bohr_si = 9.27400915E-24;
const double gamma_electron_si = 1.760859770e11;
const double boltzmann_si = 1.3806504e-23;
const double vacuum_permeadbility = 4*pi*1E-7;
const double nanometer = 1E-9;

#endif  // JAMS_CORE_CONSTS_H