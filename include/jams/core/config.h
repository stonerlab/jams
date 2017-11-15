//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>
#include "jams/core/types.h"

void config_patch(libconfig::Setting& orig, const libconfig::Setting& patch);

Vec3 config_vec3(libconfig::Setting& v);

#endif //JAMS_CONFIG_H
