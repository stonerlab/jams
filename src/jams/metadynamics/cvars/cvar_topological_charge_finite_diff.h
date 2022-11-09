// cvar_topological_charge_finite_diff.h                               -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE_FINITE_DIFF
#define INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE_FINITE_DIFF

#include <jams/metadynamics/caching_collective_variable.h>

// ***************************** WARNING *************************************
// This finite difference scheme is built on the assumption that
// you are using in-plane lattice vectors u₁ = x, u2 = x/2 + √3/2.
// NO OTHER SYSTEMS ARE SUPPORTED.
//
// (2022-06-29 Joe: in principle this can be generalised, but this is a rush
// job for Ioannis' PhD)
// ***************************** WARNING *************************************

// ---------------------------------------------------------------------------
// config settings
// ---------------------------------------------------------------------------
//
// Settings in collective_variables (standard settings are given in
// MetadynamicsPotential documentation).
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
//        name = "topological_charge_finite_diff";
//        gaussian_width = 0.05;
//        range_min = -1.2;
//        range_max = 0.2;
//        range_step = 0.01;
//      }
//    );
//  };
//

namespace jams {
    class CVarTopologicalChargeFiniteDiff : public CachingCollectiveVariable {
    public:
        CVarTopologicalChargeFiniteDiff() = default;
        explicit CVarTopologicalChargeFiniteDiff(const libconfig::Setting &settings);

        std::string name() override;

        double value() override;

        inline const jams::MultiArray<double, 2>& derivatives() override {
          throw std::runtime_error("unimplemented function");
        };

        /// Returns the value of the collective variable after a trial
        /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
        double spin_move_trial_value(
            int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

        double calculate_expensive_value() override;

    private:

        double local_topological_charge(const int i) const;
        double topological_charge_difference(int index,
                                                   const Vec3 &spin_initial,
                                                   const Vec3 &spin_final) const;

        std::string name_ = "topological_charge_finite_diff";

        // basically a CSR matrix
        std::vector<std::vector<int>> dx_indices_;
        std::vector<std::vector<double>> dx_values_;

        // basically a CSR matrix
        std::vector<std::vector<int>> dy_indices_;
        std::vector<std::vector<double>> dy_values_;
    };
}
#endif
// ----------------------------- END-OF-FILE ----------------------------------