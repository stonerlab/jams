// init_domain_wall.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_DOMAIN_WALL
#define INCLUDED_JAMS_INIT_DOMAIN_WALL

#include <jams/interface/config.h>

namespace jams {
class InitDomainWall {
public:
    /// Creates a Bloch domain wall along the x direction with the spins
    /// aligned to +/- z at the ends.
    ///
    /// Settings:
    /// center: center of the domain wall in reduced cartesian units
    /// width:  width of the domain wall in reduced cartesian units
  static void execute(const libconfig::Setting &settings);
};
}

#endif //INCLUDED_JAMS_INIT_DOMAIN_WALL
