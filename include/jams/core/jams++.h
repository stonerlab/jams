// Copyright 2017 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_JAMS_H
#define JAMS_CORE_JAMS_H
#include <cstdlib>
#include <libconfig.h++>

int jams_initialize(int argc, char **argv);
void jams_patch_config(const std::string &patch_string);
void jams_run();
void jams_finish();
void jams_error(const char *string, ...);
void jams_warning(const char *string, ...);
void jams_global_initializer(const libconfig::Setting &settings);

#endif