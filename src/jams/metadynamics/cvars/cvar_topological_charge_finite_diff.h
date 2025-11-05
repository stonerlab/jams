// cvar_topological_charge_finite_diff.h                               -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE_FINITE_DIFF
#define INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE_FINITE_DIFF

#include <jams/metadynamics/caching_collective_variable.h>
#include <jams/interface/config.h>
#include <jams/containers/vector_set.h>

// ***************************** WARNING *************************************
// This finite difference scheme is built on the assumption that
// you are using either a hexagonal lattice with in-plane lattice vectors
// u₁ = x, u₂ = x/2 + √3/2 or a square lattice with u₁ = x, u₂ = y.
// NO OTHER SYSTEMS ARE SUPPORTED.
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
    struct FDWeights {
        int index;
        double wx;
        double wy;
    };

    class CVarTopologicalChargeFiniteDiff : public CachingCollectiveVariable {
    public:
        CVarTopologicalChargeFiniteDiff() = default;
        explicit CVarTopologicalChargeFiniteDiff(const libconfig::Setting &settings);

        std::string name() override;

        double value() override;

        /// Returns the value of the collective variable after a trial
        /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
        double spin_move_trial_value(
            int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

        void spin_move_accepted(int i, const Vec3 &spin_initial,
                    const Vec3 &spin_trial) override;

        double calculate_expensive_value() override;

    private:

        double local_topological_charge(const int i) const;
        double local_topological_charge_difference_for_site(
        const std::vector<jams::FDWeights> &weights,
        int site_index,
        int moving_index,
        const Vec3 &spin_initial,
        const Vec3 &spin_final) const;
        double topological_charge_difference(int index,
                                                   const Vec3 &spin_initial,
                                                   const Vec3 &spin_final) const;

        std::string name_ = "topological_charge_finite_diff";

        // In principle the dx and dy indices can be different, but we need
        // to know all stencil points belonging to a site to be updated.
        // Here the outer vector indexes the spin 'i' sites and the inner
        // VectorSet will store a list of unique neighbour indices.
        std::vector<jams::VectorSet<int>> stencil_neighbour_indices_;

        // basically a CSR matrix
        std::vector<std::vector<FDWeights>> fd_weights_;
        std::vector<std::vector<int>> dx_indices_;
        std::vector<std::vector<double>> dx_values_;

        // basically a CSR matrix
        std::vector<std::vector<int>> dy_indices_;
        std::vector<std::vector<double>> dy_values_;

            // ---------------- Per-site caches and trial delta buffer ----------------
            // Local cached contributions: q_site_[j] = s_j · (dsx_site_[j] × dsy_site_[j])
            std::vector<double> q_site_;
            std::vector<Vec3>   dsx_site_;
            std::vector<Vec3>   dsy_site_;

            // For each moving index k, sites affected are {k} ∪ stencil_neighbour_indices_[k]
            std::vector<std::vector<int>> affected_sites_;

            struct SiteDelta {
                int   j;     // site index
                Vec3  ddsx;  // delta dsx at site j
                Vec3  ddsy;  // delta dsy at site j
                double dq;   // delta local topological charge at site j (raw, before 1/(4π))
            };
            mutable std::vector<SiteDelta> trial_deltas_;
            mutable int  trial_moving_index_ = -1;
            mutable Vec3 trial_spin_initial_ = {};
            mutable Vec3 trial_spin_final_   = {};
    };
}
#endif
// ----------------------------- END-OF-FILE ----------------------------------