// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include <libconfig.h++>

#include "jams/core/error.h"
#include "jams/core/globals.h"
#include "jams/core/utils.h"
#include "jams/core/hamiltonian.h"

#include "jams/hamiltonian/dipole.h"
#include "jams/hamiltonian/uniaxial.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_neartree.h"
#include "jams/hamiltonian/zeeman.h"

Hamiltonian* Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size) {
    // debugging output
    ::output->write("\ncreating '%s' hamiltonian\n", settings["module"].c_str());

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
