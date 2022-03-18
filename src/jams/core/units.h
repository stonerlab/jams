//
// Created by Joe Barker on 2018/09/11.
//

#ifndef JAMS_UNITS_H
#define JAMS_UNITS_H

#include <map>

#include "jams/helpers/consts.h"

namespace jams {
    const std::map<std::string, double> internal_energy_unit_conversion = {
            {"joules", kJoule2meV},
            {"J", kJoule2meV},
            {"milli_electron_volts", 1.0},
            {"meV", 1.0},
            {"milli_rydbergs", kmRyd2meV},
            {"mRyd", kmRyd2meV},
            {"rydbergs", kmRyd2meV * 1e3},
            {"Ryd", kmRyd2meV * 1e3}
    };
}

#endif //JAMS_UNITS_H
