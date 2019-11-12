//
// Created by Joe Barker on 2017/11/16.
//

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include "jams/core/jams++.h"
#include "jams/core/globals.h"

using namespace std;

void jams_die(const char *message, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, message);
  vsprintf(buffer, message, args);
  va_end(args);

  cerr << "\n********************************************************************************\n\n";
  cerr << "ERROR: " << buffer << "\n\n";
  cerr << "********************************************************************************\n\n";

  jams::cleanup_simulation();
  exit(EXIT_FAILURE);
}

void jams_warning(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  cerr << "\n********************************************************************************\n\n";
  cerr << "WARNING: " << buffer << "\n\n";
  cerr << "********************************************************************************\n\n";
}