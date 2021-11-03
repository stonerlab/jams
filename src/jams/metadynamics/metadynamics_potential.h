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
        static const int kMaxDimensions = 2;

        // TODO: Implement MirrorBCs
        enum class PotentialBCs {
          MirrorBC, // Gaussians are inserted as if the end of the ranges are mirrors.
          HardBC    // Gaussians cannot be inserted outside of the range.
        };

        MetadynamicsPotential() = default;
        explicit MetadynamicsPotential(const libconfig::Setting &settings);

        /// Inserts a Gaussian energy peak into the potential energy landscape.
        /// This may be multi-dimensional. The widths and absolute amplitude are
        /// constant and should be specified in the constructor. The relative
        /// amplitude is for scaling inserted Gaussians, for example when doing
        /// tempered metadynamics.
        void insert_gaussian(const double& relative_amplitude = 1.0);

        /// Output the potential landscape to a file stream.
        void output();

        /// Returns the value of the potential at the current coordinates of the
        /// collective variable
        double current_potential();

        /// Returns the value of the potential at the given coordinates using
        /// (bi)linear interpolation
        double potential(
            const std::array<double, kMaxDimensions>& cvar_coordinates);

        /// Calculate the difference in potential energy for the system when a
        /// single spin is changed from spin_initial to spin_final
        double potential_difference(
            int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void spin_update(
            int i, const Vec3 &spin_initial, const Vec3 &spin_final);

    private:
        const double kHardBCsPotential = 1e100; // a very large value in meV

        double              gaussian_amplitude_;
        std::vector<double> gaussian_width_;

        int                                              num_cvars_;
        std::vector<std::unique_ptr<CollectiveVariable>> cvars_;
        std::vector<std::string>                         cvar_names_;
        std::vector<PotentialBCs>                        cvar_bcs_;
        std::vector<std::vector<double>>                 cvar_sample_points_;
        std::ofstream                                    cvar_file_;

        std::array<int,kMaxDimensions>    num_samples_;
        MultiArray<double,kMaxDimensions> potential_;

    };
}

#endif //JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H
