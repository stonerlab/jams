//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_DEFAULTS_H
#define JAMS_DEFAULTS_H

#include "jams/core/types.h"

namespace jams {
    namespace defaults {
        constexpr bool   sim_verbose_output = false;

        constexpr auto   physics_module = "empty";

        constexpr auto   energy_unit_name = "joules";

        constexpr int    monitor_output_steps = 100;
        constexpr double monitor_convergence_tolerance = 0.01;

        constexpr int    solver_min_steps = 0;
        constexpr double solver_monte_carlo_move_sigma = 0.5;
        constexpr double solver_monte_carlo_constraint_tolerance = 1e-8;
        constexpr auto   solver_gpu_thermostat = "langevin-white-gpu";

        constexpr double material_gyro = 1.0;
        constexpr double material_alpha = 0.01;
        constexpr Vec3   material_spin = {0.0, 0.0, 1.0};
        constexpr Mat3   material_spin_transform = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

        constexpr bool   unitcell_symops = true;

        constexpr Vec3b  lattice_periodic_boundaries = {true, true, true};
        constexpr double lattice_tolerance = 1e-4; // default tolerance for checking distances in units of lattice constant

        constexpr int    warning_unitcell_symops_size = 100;

    } // namespace defaults
} // namespace jams

#endif //JAMS_DEFAULTS_H
