#ifndef JAMS_METADYNAMICS_MAGNETISATION_CV_H
#define JAMS_METADYNAMICS_MAGNETISATION_CV_H

#include <jams/metadynamics/collective_variable_potential.h>
#include "fstream"
#include <jams/interface/config.h>

namespace jams {
    ///
    /// Implements m_z as a collective variable for metadynamics
    ///
    class MagnetisationCollectiveVariable : public CollectiveVariablePotential {
    public:
        MagnetisationCollectiveVariable();

        MagnetisationCollectiveVariable(const libconfig::Setting &settings);

        void insert_gaussian(const double &relative_amplitude) override;

        void output() override;

        double current_potential() override;

        double potential_difference(int i, const Vec3 &spin_initial,
                                    const Vec3 &spin_final) override;

        void spin_update(int i, const Vec3 &spin_initial,
                         const Vec3 &spin_final) override;

    private:
        double collective_variable();

        double interpolated_potential(const double &value);

        double histogram_energy_difference();

        Vec3 calculate_total_magnetisation();

        Vec3 magnetisation_;

        double gaussian_amplitude_;
        double gaussian_width_;
        double histogram_step_size_;

        int lower_limit_index = 0;
        int upper_limit_index = 0;

        std::vector<double> sample_points_;
        std::vector<double> potential_;

        std::ofstream potential_difference_output_file_;

    };
}

#endif //JAMS_METADYNAMICS_MAGNETISATION_CV_H
