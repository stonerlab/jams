#ifndef JAMS_METADYNAMICS_POTENTIAL_H
#define JAMS_METADYNAMICS_POTENTIAL_H

#include <jams/containers/multiarray.h>
#include <jams/interface/config.h>
#include <jams/metadynamics/collective_variable.h>

#include <array>
#include <fstream>


namespace jams {
    class MetadynamicsPotential {
    public:
        // Maximum number of CVar dimensions supported by the class
        static constexpr int kNumCVars = 2;

        ///
        /// Boundary condition types of metadynamics CV axes.
        ///
        /// Controls how the metadynamics potential is evaluate and how
        /// gaussians are deposited when a CV approaches or exceeds
        /// a given boundary.
        ///
        /// **NoBC** - No boundary condition.
        /// There are no boundary conditions applied. If the CV does not
        /// have a limited domain, then behaviour when it exceeds the
        /// domain is not defined.
        ///
        /// **MirrorBC** - Reflect Gaussians about a configured threshold.
        /// The boundaries specified by thresholds `lower_mirror_bc_threshold`,
        /// `upper_mirror_bc_threshold` cause additional gaussians to be inserted
        /// as though the boundary is a mirror.
        ///
        /// **HardBC** - Hard wall: forbid outside-grid states with huge penalty.
        /// Exceeding the cvar min or max range results in a very large energy penalty,
        /// such that any move attempting to cross the boundary is effectively forbidden.
        ///
        /// **RestoringBC** - Spring-like restoring to threshold
        /// When the CV is beyond the `lower_restoring_bc_threshold` and
        /// `upper_restoring_bc_threshold` thresholds an additional spring like
        /// potential @f$ V_\mathrm{rest}(x) = \tfrac{1}{2}\,k\,(x - x_\mathrm{thr})^2 @f$
        /// is added. If the CV further exceeds the limits of the cvar range, no further
        /// gaussians are added. The potential is then then sum of the spring part and the
        /// clamped value of the potential at the nearest edge of the range.
        ///
        enum class PotentialBCs {
          NoBC,       // no boundary condition
          MirrorBC,   // extra 'virtual' Gaussians are inserted as if the end of the ranges are mirrors
          HardBC,     // inserting Gaussians outside the range is forbidden (extremely large potential energy penalty)
          RestoringBC // For values bigger than a threshold -> NO Gaussians are deposited, returns a V_{restoring}(Q(x)) potential.
        };

        MetadynamicsPotential() = default;
        explicit MetadynamicsPotential(const libconfig::Setting &settings);

        /// Inserts a Gaussian energy peak into the potential energy landscape.
        /// This may be multi-dimensional. The widths and absolute amplitude are
        /// constant and should be specified in the constructor. The relative
        /// amplitude is for scaling inserted Gaussians, for example when doing
        /// tempered metadynamics.
        void insert_gaussian(double relative_amplitude = 1.0);

        /// Output the potential landscape to a file stream.
        void output(const std::string& filename) const;

        /// Returns an array of the current CV coordinates
        std::array<double,kNumCVars> cvar_coordinates();

        /// Returns the value of the base (without boundary potentials) metadynamics
        /// potential at the given coordinates of the collective variable using
        /// interpolation on the grid
        double base_potential(const std::array<double, kNumCVars>& cvar_coordinates);

        /// Returns the value of the full (with boundary potentials) metadynamics potential
        /// at the given coordinates
        double full_potential(const std::array<double, kNumCVars>& cvar_coordinates);

        /// Calculate the difference in potential energy for the system when a
        /// single spin is changed from spin_initial to spin_final
        double potential_difference(
            int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void spin_update(
            int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        /// Merge potential with a shared potential stored in file
        void synchronise_shared_potential(const std::string& file_name);

        void print_settings() const;

    private:

        double get_base_potential_nearest_value(const std::array<double, kNumCVars> &cvar_coordinates);

        double get_base_potential_interpolated_value(const std::array<double, kNumCVars> &cvar_coordinates);

        /// Import the initial potential from file
        void import_potential(const std::string &filename); //  can handle up to two Collective Variables.

        /// This private method does the actual addition of the Gaussian to the
        /// internal potential. It is used by the public method
        /// 'insert_gaussian()', but allows the Gaussian center to be specified.
        /// We can then (for example) insert additional virtual Gaussians
        /// outside of the CV range when implementing mirror boundary conditions.
        void add_gaussian_to_landscape(const std::array<double,kNumCVars> center, MultiArray<double,kNumCVars>& landscape);

        /// Returns the lowest indices of the discrete potential grid square which contains cvar_coordinates
        std::array<int,kNumCVars> potential_grid_indices(const std::array<double, kNumCVars>& cvar_coordinates);

        // --- METADYNAMICS POTENTIAL
        /// the multidimensional metadynamics potential
        MultiArray<double,kNumCVars> metad_potential_;

        /// Changes in the potential since the last parallel file synchronisation
        MultiArray<double,kNumCVars> metad_potential_delta_;

        /// base amplitude (before any tempering) for metadynamics gaussian
        /// potentials
        double metad_gaussian_amplitude_{};

        bool do_metad_potential_interpolation_ = true;

        // --- COLLECTIVE VARIABLES
        /// vector of pointers to a CV instance for each CV dimension
        std::vector<std::unique_ptr<CollectiveVariable>> cvars_;

        /// vector of gaussian widths for each CV dimension
        std::vector<double> cvar_gaussian_widths_;

        /// vector of strings for the names of each CV
        std::vector<std::string> cvar_names_;

        /// vector of the minium of the discretised CV coordinates in each CV dimension
        std::vector<double> cvar_range_min_;

        /// vector of the maximum of the discretised CV coordinates in each CV dimension
        std::vector<double> cvar_range_max_;

        /// vector of 1/step of the discretised CV coordinates in each CV dimension
        std::vector<double> cvar_inv_step_;

        /// vector of vectors of the discrete CV sample coordinates in each CV dimension
        std::vector<std::vector<double>> cvar_sample_coordinates_;

        /// number of discrete CV samples in each CV dimension
        std::array<int,kNumCVars> num_cvar_sample_coordinates_{};

        /// number of iterations between CV outputs to file
        int cvar_output_stride_{};

        /// CV output file
        std::ofstream cvar_output_file_;


        // --- BOUNDARY CONDITIONS

        /// vector of the lower boundary condition type in each CV dimension
        std::vector<PotentialBCs> cvar_lower_bcs_;

        /// vector of the upper boundary condition type in each CV dimension
        std::vector<PotentialBCs> cvar_upper_bcs_;

        // --- --- Restoring boundary conditions

        /// lower CV coordinate at which to apply restoring boundary conditions
        std::array<double,kNumCVars> restoring_bc_lower_threshold_{};

        /// upper CV coordinate at which to apply restoring boundary conditions
        std::array<double,kNumCVars> restoring_bc_upper_threshold_{};

        /// lower CV coordinate at which to apply mirror boundary conditions
        std::array<double,kNumCVars> mirror_bc_lower_threshold_{};

        /// upper CV coordinate at which to apply mirror boundary conditions
        std::array<double,kNumCVars> mirror_bc_upper_threshold_{};

        /// restoring boundary condition spring constant
        std::array<double,kNumCVars> restoring_bc_spring_constant_{};

        // --- --- Hard boundary conditions

        /// amplitude of the large potential to apply for hard boundary
        /// conditions. Units are JAMS internal units (meV)
        const double kHardBCsPotential = 1e100;

        /// Number of Gaussian widths after which to set to zero
        const double kGaussianExtent = 3.0;
    };

    inline const char* to_string(MetadynamicsPotential::PotentialBCs bc) noexcept {
        switch (bc) {
            case MetadynamicsPotential::PotentialBCs::NoBC:    return "NoBC";
            case MetadynamicsPotential::PotentialBCs::MirrorBC:    return "MirrorBC";
            case MetadynamicsPotential::PotentialBCs::HardBC:      return "HardBC";
            case MetadynamicsPotential::PotentialBCs::RestoringBC: return "RestoringBC";
            default:                        return "UnknownBC";
        }
    }

    inline std::ostream& operator<<(std::ostream& os, MetadynamicsPotential::PotentialBCs bc) {
        return os << to_string(bc);
    }
}

#endif //JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H
