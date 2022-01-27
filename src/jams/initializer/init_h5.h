// init_h5.h                                                           -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_H5_WALL
#define INCLUDED_JAMS_INIT_H5_WALL

#include <jams/interface/config.h>

namespace jams {
class InitH5 {
public:
  static void execute(const libconfig::Setting &settings);
};
}

#endif //INCLUDED_JAMS_INIT_H5_WALL
