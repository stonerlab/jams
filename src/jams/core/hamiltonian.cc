// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include "jams/core/globals.h"
#include "jams/core/utils.h"
#include "jams/core/hamiltonian.h"

#include "jams/hamiltonian/dipole.h"
#include "jams/hamiltonian/uniaxial.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_neartree.h"
#include "jams/hamiltonian/zeeman.h"

Hamiltonian* Hamiltonian::create(const libconfig::Setting &settings) {
    // debugging output
    ::output.write("\ncreating '%s' hamiltonian\n", settings["module"].c_str());

    if (capitalize(settings["module"]) == "EXCHANGE") {
        return new ExchangeHamiltonian(settings);
    }

    if (capitalize(settings["module"]) == "EXCHANGE_NEARTREE") {
        return new ExchangeNeartreeHamiltonian(settings);
    }

    if (capitalize(settings["module"]) == "UNIAXIAL") {
        return new UniaxialHamiltonian(settings);
    }

    if (capitalize(settings["module"]) == "DIPOLE") {
        return new DipoleHamiltonian(settings);
    }

    if (capitalize(settings["module"]) == "ZEEMAN") {
        return new ZeemanHamiltonian(settings);
    }

    // throw error if the hamiltonian name is no known
    jams_error("Unknown hamiltonian name specified '%s'", settings["module"].c_str());
    return NULL;
}
