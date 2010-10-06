#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "output.h"
#include "rand.h"

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#include <libconfig.h++>
#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif


//extern Cell *cell;  ///< Computational cell object

GLOBAL libconfig::Config config;  ///< Config object

GLOBAL Output output;

GLOBAL Random rng;

//extern Solver *solver;

void jams_error(const char *string, ...);

#undef GLOBAL

#endif // __GLOBALS_H_
