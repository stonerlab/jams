#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "output.h"

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif


//extern Cell *cell;  ///< Computational cell object

//extern ConfigFile config;  ///< Config object

GLOBAL Output output;

//extern Solver *solver;


#endif // __GLOBALS_H_
