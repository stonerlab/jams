#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#include "vecfield.h"
#include "array.h"
#include "output.h"
#include "rand.h"
#include "solver.h"
//#include "geometry.h"
#include "lattice.h"
#include "sparsematrix.h"

#include <libconfig.h++>

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif


namespace globals
{
  GLOBAL int nspins;

  GLOBAL VecField<double> s;
  GLOBAL VecField<double> h;
  GLOBAL VecField<double> w;

  GLOBAL SparseMatrix * jij;

  GLOBAL Array<double> alpha;
  GLOBAL Array<double> mus;
  GLOBAL Array<double> gyro;

} // namespace global

//GLOBAL Cell *cell;  ///< Computational cell object

//GLOBAL Geometry geometry;
GLOBAL Lattice lattice;

GLOBAL libconfig::Config config;  ///< Config object

GLOBAL Output output;

GLOBAL Random rng;

void jams_error(const char *string, ...);

#undef GLOBAL

#endif // __GLOBALS_H_
