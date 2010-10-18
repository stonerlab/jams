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
#include "solver.h"
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

  GLOBAL Array2D<double> s;
  GLOBAL Array2D<double> h;
  GLOBAL Array2D<double> w;

  GLOBAL SparseMatrix<double> Jij;

  GLOBAL Array<double> alpha;
  GLOBAL Array<double> mus;
  GLOBAL Array<double> gyro;

} // namespace global



#undef GLOBAL

#endif // __GLOBALS_H_
