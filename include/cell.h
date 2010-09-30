#ifndef __CELL_H__
#define __CELL_H__

#include "global.h"
#include "vecfield.h"

class Cell
{
  public:

  /////////////////////////////////////////////////////////////////////

    int nspins; // number of spins

    vecField s; // spins
    vecField h; // internal fields
    vecField w; // Weiner processes

//    Field mus;    // moments
//    Field alpha;  // Gilbert damping
//    Field gyro;   // gyromagnetic ratio

//    sparseMatrix Jij

};

#endif // __CELL_H__
