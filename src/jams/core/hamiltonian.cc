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

using namespace std;

Hamiltonian* Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size) {
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

  throw std::runtime_error("unknown hamiltonian " + std::string(settings["module"].c_str()));
}

Hamiltonian::Hamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : energy_(size, 0.0),
          field_(size, 3, 0.0),
          name_(settings["module"].c_str())
{
  cout << "  " << name() << " hamiltonian\n";
}
