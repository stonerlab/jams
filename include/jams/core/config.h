//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>

void config_patch(libconfig::Setting& orig, const libconfig::Setting& patch);

#endif //JAMS_CONFIG_H
