// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <cstddef>
#include <iostream>
#include <libconfig.h++>

#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/core/thermostat.h"

#include "../thermostats/cuda_langevin_white.h"
#include "../thermostats/cuda_langevin_bose.h"
#include "../thermostats/cuda_langevin_arbitrary.h"

using namespace std;
Thermostat* Thermostat::create(const std::string &thermostat_name) {
  cout << thermostat_name << " thermostat\n";

    // create the selected thermostat
    #if HAS_CUDA
    if (capitalize(thermostat_name) == "LANGEVIN-WHITE-GPU" || capitalize(thermostat_name) == "CUDA_LANGEVIN_WHITE") {
        return new CudaLangevinWhiteThermostat(config->lookup("physics.temperature"), 0.0, globals::num_spins);
    }
    if (capitalize(thermostat_name) == "LANGEVIN-BOSE-GPU" ||capitalize(thermostat_name) == "CUDA_LANGEVIN_COTH") {
        return new CudaLangevinBoseThermostat(config->lookup("physics.temperature"), 0.0, globals::num_spins);
    }
    if (capitalize(thermostat_name) == "LANGEVIN-ARBITRARY-GPU" ||capitalize(thermostat_name) == "CUDA_LANGEVIN_ARBITRARY") {
      return new CudaLorentzianThermostat(config->lookup("physics.temperature"), 0.0, globals::num_spins);
    }
    #endif

    // throw error if the thermostat name is no known
  throw std::runtime_error("unknown thermostat " + thermostat_name);
}
