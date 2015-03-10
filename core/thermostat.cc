// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include "core/globals.h"
#include "core/utils.h"
#include "core/thermostat.h"

#include "thermostats/cuda_langevin_white.h"
#include "thermostats/cuda_langevin_coth.h"

Thermostat* Thermostat::create(const std::string &thermostat_name) {
    // debugging output
    if (::verbose_output_is_set) {
        ::output.write("\ncreating '%s' thermostat\n", thermostat_name.c_str());
    }

    // create the selected thermostat
    if (capitalize(thermostat_name) == "CUDA_LANGEVIN_WHITE") {
        return new CudaLangevinWhiteThermostat(0.0, 0.0, globals::num_spins);
    }
    if (capitalize(thermostat_name) == "CUDA_LANGEVIN_COTH") {
        return new CudaLangevinCothThermostat(0.0, 0.0, globals::num_spins);
    }

    // throw error if the thermostat name is no known
    jams_error("Unknown thermostat requested '%s'", thermostat_name.c_str());
    return NULL;
}
