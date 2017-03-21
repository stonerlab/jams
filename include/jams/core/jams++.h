// Copyright 2017 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_JAMS_H
#define JAMS_CORE_JAMS_H
#include <cstdlib>

int jams_initialize(int argc, char **argv);
void jams_run();
void jams_finish();
void jams_error(const char *string, ...);
void jams_warning(const char *string, ...);

#endif