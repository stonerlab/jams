// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include "core/globals.h"
#include "core/utils.h"
#include "core/hamiltonian.h"

#include "hamiltonian/dipole.h"
#include "hamiltonian/uniaxial.h"
#include "hamiltonian/exchange.h"
#include "hamiltonian/zeeman.h"

Hamiltonian* Hamiltonian::create(const libconfig::Setting &settings) {
    // debugging output
    if (::verbose_output_is_set) {
        ::output.write("\ncreating '%s' hamiltonian\n", settings["module"].c_str());
    }

    if (capitalize(settings["module"]) == "EXCHANGE") {
        return new ExchangeHamiltonian(settings);
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
