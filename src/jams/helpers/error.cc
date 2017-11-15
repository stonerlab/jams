//
// Created by Joe Barker on 2017/11/16.
//

#include <cstdarg>
#include <cstdio>
#include "jams/core/jams++.h"
#include "jams/core/globals.h"
#include "jams/core/output.h"

void jams_error(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output->write("\n********************************************************************************\n\n");
  output->write("ERROR: %s\n\n", buffer);
  output->write("********************************************************************************\n\n");

  jams_finish();
  exit(EXIT_FAILURE);
}

void jams_warning(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output->write("\n********************************************************************************\n\n");
  output->write("WARNING: %s\n\n", buffer);
  output->write("********************************************************************************\n\n");
}