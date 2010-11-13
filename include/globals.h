#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

void jams_error(const char *string, ...);

#include "output.h"
#include "rand.h"
#include "lattice.h"

#include <libconfig.h++>

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif

GLOBAL Lattice lattice;

GLOBAL libconfig::Config config;  ///< Config object

GLOBAL Output output;

GLOBAL Random rng;

#include "array.h"
#include "array2d.h"
#include "sparsematrix.h"

namespace globals
{
  GLOBAL int nspins;
  GLOBAL int nspins3;

  GLOBAL double h_app[3];

  GLOBAL Array2D<double> s;
  GLOBAL Array2D<double> h;
  GLOBAL Array2D<double> w;

  GLOBAL SparseMatrix<double> Jij;

  GLOBAL Array<double> alpha;
  GLOBAL Array<double> mus;
  GLOBAL Array<double> gyro;
  GLOBAL Array<double> omega_corr;

} // namespace global



#undef GLOBAL

#endif // __GLOBALS_H_
