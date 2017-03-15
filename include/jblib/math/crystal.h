// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JB_MATH_CRYSTAL_H
#define JB_MATH_CRYSTAL_H

#include <ostream>

#include "jblib/sys/types.h"


namespace jblib {
  enum crystal_t {
    cubic,
    tetragonal,
    orthorhombic,
    monoclinic,
    triclinic,
    trigonal,
    hexagonal,
    undefined
  };

  std::ostream& operator<<(std::ostream& os, enum crystal_t x);

  crystal_t identify_crystal_system(const float64 a[3], const float64 b[3], const float64 c[3]);
}  // namespace jblib

#endif
