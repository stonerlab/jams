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

  GLOBAL SparseMatrix * jijxx;
  GLOBAL SparseMatrix * jijxy;
  GLOBAL SparseMatrix * jijxz;
  GLOBAL SparseMatrix * jijyx;
  GLOBAL SparseMatrix * jijyy;
  GLOBAL SparseMatrix * jijyz;
  GLOBAL SparseMatrix * jijzx;
  GLOBAL SparseMatrix * jijzy;
  GLOBAL SparseMatrix * jijzz;

  GLOBAL Array<double> alpha;
  GLOBAL Array<double> mus;
  GLOBAL Array<double> gyro;

} // namespace global

GLOBAL Lattice lattice;

GLOBAL libconfig::Config config;  ///< Config object

GLOBAL Output output;

GLOBAL Random rng;

void jams_error(const char *string, ...);

#undef GLOBAL

#endif // __GLOBALS_H_
