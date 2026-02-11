//
// Created by Joe Barker on 2017/11/16.
//

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include "jams/helpers/utils.h"

void jams_warning(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsnprintf(buffer, sizeof(buffer), string, args);
  va_end(args);

  std::cerr << "\n********************************************************************************\n";
  std::cerr << "WARNING: \n";
  std::cerr << string_wrap(buffer, 80);
  std::cerr << "\n";
  std::cerr << "********************************************************************************\n";
  std::cerr << std::flush;
}
