// cvar_magnetisation.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_MAGNETISATION
#define INCLUDED_JAMS_CVAR_MAGNETISATION

#include <jams/metadynamics/caching_collective_variable.h>

#include <jams/containers/vec3.h>

#include <vector>

// ---------------------------------------------------------------------------
// config settings
// ---------------------------------------------------------------------------
//
// Settings in collective_variables (standard settings are given in
// MetadynamicsPotential documentation).
//
// component : (string) The component of magnetisation to use (x, y, z).
//
// -------
//
//  solver : {
//    module = "monte-carlo-metadynamics-cpu";
//    max_steps  = 100000;
//    gaussian_amplitude = 10.0;
//    gaussian_deposition_stride = 100;
//    output_steps = 100;
//
//    collective_variables = (
//      {
//        name = "magnetisation";
//        component = "z";
//        gaussian_width = 0.01;
//        range_min = 0.0;
//        range_max = 1.0;
//        range_step = 0.001;
//      }
//    );
//  };
//

namespace jams {
class CVarMagnetisation : public CachingCollectiveVariable {
public:
    CVarMagnetisation() = default;
    explicit CVarMagnetisation(const libconfig::Setting &settings);

    std::string name() override;

    double value() override;

    /// Returns the value of the collective variable after a trial
    /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
    double spin_move_trial_value(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

    double calculate_expensive_value() override;

private:
    std::string name_ = "magnetisation";
    std::string material_;
    int magnetisation_component_;

};
}


#endif //INCLUDED_JAMS_CVAR_MAGNETISATION
