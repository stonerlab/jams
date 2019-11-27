//
// Created by Joseph Barker on 2019-11-12.
//

#ifndef JAMS_INTERACTION_CALCULATOR_H
#define JAMS_INTERACTION_CALCULATOR_H

#include "jams/helpers/output.h"
#include "jams/core/lattice.h"
#include "jams/core/interactions.h"

namespace jams {
    void interaction_calculator(const Cell& unitcell, const std::vector<Atom>& motif, double r_max, double eps = 1e-5);
}

#endif //JAMS_INTERACTION_CALCULATOR_H
