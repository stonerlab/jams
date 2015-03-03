// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include "core/globals.h"
#include "core/thermostat.h"

#include "thermostats/cuda_langevin_white.h"

Thermostat* Thermostat::create(const std::string &thermostat_name) {
    // debugging output
    if (::verbose_output_is_set) {
        ::output.write("\ncreating '%s' thermostat\n", thermostat_name.c_str());
    }

    // create the selected thermostat
    if (thermostat_name == "CUDA_LANGEVIN_WHITE") {
        return new CudaLangevinWhiteThermostat(0.0, 0.0, 0);
    }
    // if (thermostat_name == "LANGEVIN_COTH") {
    //     return new LangevinCothThermostat();
    // }

    // throw error if the thermostat name is no known
    jams_error("Unknown thermostat requested '%s'", thermostat_name.c_str());
    return NULL;
}
