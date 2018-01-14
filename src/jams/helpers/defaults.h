//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_DEFAULTS_H
#define JAMS_DEFAULTS_H

#include "jams/core/types.h"

namespace jams {
    constexpr bool   default_sim_verbose_output = false;

    constexpr auto default_physics_module = "empty";

    constexpr int    default_monitor_output_steps = 100;
    constexpr double default_monitor_convergence_tolerance = 0.01;

    constexpr int    default_solver_min_steps = 0;
    constexpr double default_solver_monte_carlo_move_sigma = 0.5;
    constexpr auto   default_solver_gpu_thermostat = "langevin-white-gpu";

    constexpr double default_material_gyro = 1.0;
    constexpr double default_material_alpha = 0.01;
    constexpr Vec3   default_material_spin = {0.0, 0.0, 1.0};
    constexpr Mat3   default_material_spin_transform = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    constexpr bool   default_unitcell_symops = true;

    constexpr Vec3b  default_lattice_periodic_boundaries = {true, true, true};

}

#endif //JAMS_DEFAULTS_H
