//
// Created by Joe Barker on 2018/09/11.
//

#ifndef JAMS_UNITS_H
#define JAMS_UNITS_H

#include <map>

#include "jams/helpers/consts.h"

namespace jams {
    const std::map<std::string, double> internal_energy_unit_conversion = {
            {"joules", 1.0/kBohrMagneton},
            {"milli_electron_volts", 1.6021766e-22/kBohrMagneton},
            {"milli_rydbergs", 2.1798724e-21/kBohrMagneton},
            {"rydbergs", 2.1798724e-18/kBohrMagneton}
    };
}

#endif //JAMS_UNITS_H
