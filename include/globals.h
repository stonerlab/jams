#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "output.h"
#include "config.h"

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif


//extern Cell *cell;  ///< Computational cell object

GLOBAL Config config;  ///< Config object

GLOBAL Output output;

//extern Solver *solver;

void jams_error(const char *string, ...);

#undef GLOBAL

#endif // __GLOBALS_H_
