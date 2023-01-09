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

        enum class PotentialBCs {
          MirrorBC, // Gaussians are inserted as if the end of the ranges are mirrors.
          HardBC,    // Gaussians cannot be inserted outside of the range.
          RestoringBC // For values bigger than a threshold -> NO Gaussians are deposited, returns a V_{restoring}(Q(x)) potential.
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
        void import_potential(const std::string &filename); //  can handle up to two Collective Variables.

        /// This private method does the actual addition of the Gaussian to the
        /// internal potential. It is used by the public method
        /// 'insert_gaussian()', but allows the Gaussian center to be specified.
        /// We can then (for example) insert additional virtual Gaussians
        /// outside of the CV range when implementing mirror boundary conditions.
        void add_gaussian_to_potential(const double relative_amplitude, const std::array<double,kMaxDimensions> center);


        double              gaussian_amplitude_;
        std::vector<double> gaussian_width_;
        double lower_restoringBC_threshold_;
        double upper_restoringBC_threshold_;
        double restoringBC_string_constant_;
        bool   potential_input_file = false;
        int cvar_file_output_;

        int                                              num_cvars_; //used to resize all the other vectors
        std::vector<std::unique_ptr<CollectiveVariable>> cvars_;
        std::vector<std::string>                         cvar_names_;
        std::vector<PotentialBCs>                        lower_cvar_bc_;
        std::vector<PotentialBCs>                        upper_cvar_bc_;
        std::vector<double>                              cvar_range_min_;
        std::vector<double>                              cvar_range_max_;
        std::vector<std::vector<double>>                 cvar_sample_points_;
        std::ofstream                                    cvar_file_;

        std::array<int,kMaxDimensions>    num_samples_;
        MultiArray<double,kMaxDimensions> potential_;

    };
}

#endif //JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H
