//
// Created by Joe Barker on 2017/11/16.
//

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include "jams/core/jams++.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"

void jams_die(const char *message, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, message);
  vsprintf(buffer, message, args);
  va_end(args);

  std::cerr << "\n********************************************************************************\n\n";
  std::cerr << "ERROR: " << buffer << "\n\n";
  std::cerr << "********************************************************************************\n\n";

  jams::cleanup_simulation();
  exit(EXIT_FAILURE);
}

void jams_warning(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  std::cerr << "\n********************************************************************************\n";
  std::cerr << "WARNING: \n";
  std::cerr << string_wrap(buffer, 80);
  std::cerr << "\n";
  std::cerr << "********************************************************************************\n";
  std::cerr << std::flush;
}