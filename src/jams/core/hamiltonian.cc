// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/units.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"

#include "jams/hamiltonian/dipole.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_neartree.h"
#include "jams/hamiltonian/random_anisotropy.h"
#include "jams/hamiltonian/uniaxial_anisotropy.h"
#include "jams/hamiltonian/cubic_anisotropy.h"
#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"
#include "jams/hamiltonian/zeeman.h"

#if HAS_CUDA
  #include "jams/hamiltonian/cuda_exchange.h"
  #include "jams/hamiltonian/cuda_exchange_neartree.h"
  #include "jams/hamiltonian/cuda_random_anisotropy.h"
  #include "jams/hamiltonian/cuda_uniaxial_anisotropy.h"
  #include "jams/hamiltonian/cuda_cubic_anisotropy.h"
  #include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h"
  #include "jams/hamiltonian/cuda_zeeman.h"
#endif

using namespace std;

Hamiltonian * Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size, bool is_cuda_solver) {
    if (capitalize(settings["module"]) == "EXCHANGE") {
        #if HAS_CUDA
          if (is_cuda_solver) {
            return new CudaExchangeHamiltonian(settings, size);
          }
        #endif
        return new ExchangeHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "EXCHANGE-NEARTREE") {
        #if HAS_CUDA
          if (is_cuda_solver) {
            return new CudaExchangeNeartreeHamiltonian(settings, size);
          }
        #endif
        return new ExchangeNeartreeHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "UNIAXIAL") {
      #if HAS_CUDA
      if (is_cuda_solver) {
        return new CudaUniaxialHamiltonian(settings, size);
      }
      #endif
        return new UniaxialHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "UNIAXIAL-MICRO") {
    #if HAS_CUDA
        if (is_cuda_solver) {
          return new CudaUniaxialMicroscopicHamiltonian(settings, size);
        }
    #endif
        return new UniaxialMicroscopicHamiltonian(settings, size);
      }

    if (capitalize(settings["module"]) == "CUBIC") {
        #if HAS_CUDA
        if (is_cuda_solver) {
        return new CudaCubicHamiltonian(settings, size);
      }
        #endif
        return new CubicHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "DIPOLE") {
        return new DipoleHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "ZEEMAN") {
#if HAS_CUDA
      if (is_cuda_solver) {
        return new CudaZeemanHamiltonian(settings, size);
      }
#endif
      return new ZeemanHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "RANDOM-ANISOTROPY") {
#if HAS_CUDA
      if (is_cuda_solver) {
        return new CudaRandomAnisotropyHamiltonian(settings, size);
      }
#endif
      return new RandomAnisotropyHamiltonian(settings, size);
    }

  throw std::runtime_error("unknown hamiltonian " + std::string(settings["module"].c_str()));
}

Hamiltonian::Hamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Base(settings),
          energy_(size),
          field_(size, 3),
          name_(settings["module"].c_str()),
          input_unit_name_(jams::config_optional<string>(settings, "unit_name", jams::defaults::energy_unit_name))
{
  cout << "  " << name() << " hamiltonian\n";

  if (!jams::internal_energy_unit_conversion.count(input_unit_name_)) {
    throw runtime_error("units: " + input_unit_name_ + " is not known");
  }

  input_unit_conversion_ = jams::internal_energy_unit_conversion.at(input_unit_name_);
}
