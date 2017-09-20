//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>

void replace_settings(libconfig::Setting& original, const libconfig::Setting& additions, const bool& allow_type_change);

#endif //JAMS_CONFIG_H
