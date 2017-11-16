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

Hamiltonian* Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size) {
    if (capitalize(settings["module"]) == "EXCHANGE") {
        return new ExchangeHamiltonian(settings, size);
    }

    if (capitalize(settings["module"]) == "EXCHANGE_NEARTREE") {
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

    // throw error if the hamiltonian name is no known
    jams_error("Unknown hamiltonian name specified '%s'", settings["module"].c_str());
    return nullptr;
}
