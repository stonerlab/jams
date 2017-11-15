//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_DEFAULTS_H
#define JAMS_DEFAULTS_H

#include "jams/core/types.h"

namespace jams {

    // materials
    constexpr double default_gyro = 1.0;
    constexpr double default_alpha = 0.01;
    constexpr Vec3   default_spin = {0.0, 0.0, 1.0};
    constexpr Vec3   default_spin_transform = {1.0, 1.0, 1.0};

    // lattice
    constexpr Vec3b  default_lattice_periodic_boundaries = {true, true, true};
    // modules
    constexpr auto default_physics_module = "empty";
}

#endif //JAMS_DEFAULTS_H
