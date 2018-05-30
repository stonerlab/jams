// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include <libconfig.h++>

#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "hamiltonian.h"

#include "jams/hamiltonian/dipole.h"
#include "jams/hamiltonian/anisotropy_uniaxial.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_neartree.h"
#include "jams/hamiltonian/zeeman.h"
#include "jams/hamiltonian/random_anisotropy.h"
#include "jams/hamiltonian/random_anisotropy_cuda.h"

using namespace std;

Hamiltonian * Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size, bool is_cuda_solver) {
    if (capitalize(settings["module"]) == "EXCHANGE") {
        return new ExchangeHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "EXCHANGE-NEARTREE") {
        return new ExchangeNeartreeHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "UNIAXIAL") {
        return new UniaxialHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "DIPOLE") {
        return new DipoleHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "ZEEMAN") {
        return new ZeemanHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "RANDOM-ANISOTROPY") {
      if (is_cuda_solver) {
        return new RandomAnisotropyCudaHamiltonian(settings, size);
      }
      return new RandomAnisotropyHamiltonian(settings, size);
    }

  throw std::runtime_error("unknown hamiltonian " + std::string(settings["module"].c_str()));
}

Hamiltonian::Hamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Base(settings),
          energy_(size, 0.0),
          field_(size, 3, 0.0),
          name_(settings["module"].c_str())
{
  cout << "  " << name() << " hamiltonian\n";
}
