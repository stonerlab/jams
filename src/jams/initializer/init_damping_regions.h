// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef INCLUDED_JAMS_INIT_DAMPING_REGIONS
#define INCLUDED_JAMS_INIT_DAMPING_REGIONS

#include <jams/interface/config.h>

namespace jams {

class InitDampingRegions {
 public:
  static void execute(const libconfig::Setting& settings);
};

}  // namespace jams

#endif  // INCLUDED_JAMS_INIT_DAMPING_REGIONS
