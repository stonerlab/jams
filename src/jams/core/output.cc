// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/output.h"

#include <cstdarg>
#include <cstdio>
#include <string>

void Output::open(const char *fname, ...) {
  va_list args;

  if (fname == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args, fname);
    vsprintf(buffer, fname, args);
  va_end(args);


  logfile.open(buffer);

  if (!logfile.is_open()) {
    fprintf(stderr, "Failed to open log file %s\n", buffer);
  }
}

void Output::close() {
  logfile.close();
}

void Output::write(const char* message, ...) {
  va_list args;

  if (message == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args, message);
    vsprintf(buffer, message, args);
  va_end(args);

  logfile << buffer;
  logfile.flush();

  if (console == true) {
      print(buffer);
      fflush(stdout);
  }
}

void Output::verbose(const char* message, ...) {
  if (verbose_) {
    va_list args;

    if (message == reinterpret_cast<const char*>(NULL)) {
      return;
    }

    va_start(args, message);
      vsprintf(buffer, message, args);
    va_end(args);

    logfile << buffer;
    logfile.flush();

    if (console == true) {
        print(buffer);
        fflush(stdout);
    }
  }
}

void Output::print(const char* message, ...) {
  va_list args;

  if (message == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args, message);
    vprintf(message, args);
  va_end(args);

  fflush(stdout);
}