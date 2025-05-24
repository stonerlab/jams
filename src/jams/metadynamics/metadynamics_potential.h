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

        void initialise_shared_potential_file(const std::string& file_name);

        /// Merge potential with a shared potential stored in file
        void synchronise_shared_potential(const std::string& file_name);

    private:
        /// Import the initial potential from file
        void import_potential(const std::string &filename); //  can handle up to two Collective Variables.

        /// This private method does the actual addition of the Gaussian to the
        /// internal potential. It is used by the public method
        /// 'insert_gaussian()', but allows the Gaussian center to be specified.
        /// We can then (for example) insert additional virtual Gaussians
        /// outside of the CV range when implementing mirror boundary conditions.
        void add_gaussian_to_potential(const double relative_amplitude, const std::array<double,kMaxDimensions> center);

        // --- METADYNAMICS POTENTIAL
        /// the multidimensional metadynamics potential
        MultiArray<double,kMaxDimensions> metad_potential_;

        /// Changes in the potential since the last parallel file synchronisation
        MultiArray<double,kMaxDimensions> metad_potential_delta_;

        /// base amplitude (before any tempering) for metadynamics gaussian
        /// potentials
        double metad_gaussian_amplitude_;

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

        /// vector of vectors of the discrete CV sample coordinates in each CV dimension
        std::vector<std::vector<double>> cvar_sample_coordinates_;

        /// number of discrete CV samples in each CV dimension
        std::array<int,kMaxDimensions> num_cvar_sample_coordinates_;

        /// number of iterations between CV outputs to file
        int cvar_output_stride_;

        /// CV output file
        std::ofstream cvar_output_file_;


        // --- BOUNDARY CONDITIONS

        /// vector of the lower boundary condition type in each CV dimension
        std::vector<PotentialBCs> cvar_lower_bcs_;

        /// vector of the upper boundary condition type in each CV dimension
        std::vector<PotentialBCs> cvar_upper_bcs_;

        // --- --- Restoring boundary conditions

        /// lower CV coordinate at which to apply restoring boundary conditions
        double restoring_bc_lower_threshold_;

        /// upper CV coordinate at which to apply restoring boundary conditions
        double restoring_bc_upper_threshold_;

        /// restoring boundary condition spring constant
        double restoring_bc_spring_constant_;

        // --- --- Hard boundary conditions

        /// amplitude of the large potential to apply for hard boundary
        /// conditions. Units are JAMS internal units (meV)
        const double kHardBCsPotential = 1e100;


    };
}

#endif //JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H
