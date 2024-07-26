// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_ERROR_H
#define JAMS_CORE_ERROR_H

#include <jams/core/jams++.h>
#include <cstdlib>
#include <iostream>
#include <libconfig.h>

namespace jams {
  template <class ... Args>
  void die(const char* reason, Args && ... details) {

    std::cerr << "\n" << reason << "\n";
    ([&]{
      std::cerr << details;
    } (), ...);
    std::cerr << std::flush;

    jams::cleanup_simulation();
    std::exit(EXIT_FAILURE);
  }
}

void jams_warning(const char *message, ...);

#define Q(x) #x
#define QUOTE(x) Q(x)

#define LINE_STRING QUOTE(__LINE__)

#define JAMS_FUNCTION()

#define JAMS_FILE __FILE__
#define JAMS_FILE_LINE JAMS_FILE ":" LINE_STRING


#define JAMS_ERROR_MESSAGE(msg) (  \
  std::string(JAMS_FILE_LINE ": error: " msg)  \
  )

#endif
