#ifndef __CELL_H__
#define __CELL_H__

#include "globals.h"
#include "vecfield.h"

class Cell
{
  public:

  /////////////////////////////////////////////////////////////////////

    int nspins; // number of spins

    vecField<double> s; // spins
    vecField<double> h; // internal fields
    vecField<double> w; // Weiner processes

//    Field mus;    // moments
//    Field alpha;  // Gilbert damping
//    Field gyro;   // gyromagnetic ratio

//    sparseMatrix Jij

};

#endif // __CELL_H__
