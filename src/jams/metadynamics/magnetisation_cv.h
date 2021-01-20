#ifndef JAMS_METADYNAMICS_MAGNETISATION_CV_H
#define JAMS_METADYNAMICS_MAGNETISATION_CV_H

#include <jams/metadynamics/collective_variable_potential.h>
#include "fstream"
#include <jams/interface/config.h>

namespace jams {
    class MagnetisationCollectiveVariable : public CollectiveVariablePotential {
    public:
        MagnetisationCollectiveVariable();
        MagnetisationCollectiveVariable(const libconfig::Setting& settings);

        void insert_gaussian(const double& relative_amplitude) override;

        void output() override;

        double potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

        void spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

    private:
        // in this case this is 1D but could be multi-dimensional
        /// Returns the current value of the collective coordinate
        double collective_coordinate();

        double amplitude_tempering(const double m);

        double interpolated_potential(const double &value);

        double histogram_energy_difference();

	    void physical_region_indices (); //find the physical limits [-1,1]

        Vec3 calculate_total_magnetisation();

        Vec3 magnetisation_;

        double gaussian_amplitude_;
        double gaussian_width_;
        double histogram_step_size_;
        bool tempering_ ;
        double bias_temperature_ ;
	    double gaussian_amplitude_used;
	    std::string sim_type_selected;

	    double lower_limit_index = 0.0;
	    double upper_limit_index = 0.0;

        std::vector<double> sample_points_;
        std::vector<double> potential_;

	  std::ofstream potential;
	  std::ofstream metadynamics_simulation_parameters;

    };
}

#endif //JAMS_METADYNAMICS_MAGNETISATION_CV_H
