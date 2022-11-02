// initializer_dispatcher.h                                               -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_DISPATCHER
#define INCLUDED_JAMS_INIT_DISPATCHER

#include <jams/interface/config.h>

namespace jams {
  class InitializerDispatcher {
  public:
      static void execute(const libconfig::Setting &settings);
  };
}

#endif //INCLUDED_JAMS_INIT_DISPATCHER
