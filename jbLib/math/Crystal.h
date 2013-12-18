#ifndef JB_CRYSTAL_H
#define JB_CRYSTAL_H

#include <ostream>

namespace jbLib{
  enum crystal_t { cubic, tetragonal, orthorhombic, monoclinic, triclinic, trigonal, hexagonal, undefined};

  std::ostream& operator<<(std::ostream& os, enum crystal_t x);

  namespace jbCrystal {
    crystal_t identifyCrystalSystem(const double a[3], const double b[3], const double c[3]);

  };
}

#endif
