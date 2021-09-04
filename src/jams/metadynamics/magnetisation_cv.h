#ifndef JAMS_METADYNAMICS_MAGNETISATION_CV_H
#define JAMS_METADYNAMICS_MAGNETISATION_CV_H

#include <jams/metadynamics/collective_variable_potential.h>
#include "fstream"
#include <jams/interface/config.h>

namespace jams {
    ///
    /// Implements m_z as a collective variable for metadynamics
    ///
    /// The collective variable is
    ///
    /// $$m_z = \frac{1}{N} \sum_{i=0}{N} \vec{S}_i$$
    ///
    /// where N is the number of spins.
    ///
    /// WARNING: The sum does not include the magnetic moment so it is not
    /// necessarily suitable for systems with spins of different size.
    ///
    ///-------------------------------------------------------------------------
    /// Config settings
    ///-------------------------------------------------------------------------
    ///
    /// For collective variables these appear in the "solver" group:
    ///
    /// - collective_variable = "magnetisation"
    ///
    /// Selects this collective variable for use with the metadynamics
    ///
    /// - gaussian_amplitude = float;
    ///
    /// Amplitude of the Gaussian inserted into the potential in units of
    /// Joules. Note that the energy barriers calculated in metadynamics are for
    /// the whole system, not per spin. So gaussian_amplitude should be relative
    /// to the global energy landscape not per spin.
    ///
    /// - gaussian_width = float;
    ///
    /// Width of the Gaussian inserted into the potential in units of m_z. This
    /// probably should be of the order of 0.02 - 0.10.
    ///
    /// - histogram_step_size = float;
    ///
    /// Step size for discretising the potential landscape in units of m_z. The
    /// smaller it is the finer the mesh but the more expensive some calls are.
    /// Sensible values are of the order 0.02. NOTE: the value must divide
    /// exactly into 2.0 so that it includes the -1 and 1 as end points.
    ///
    ///-------------------------------------------------------------------------
    /// Implementation details
    ///-------------------------------------------------------------------------
    ///
    /// We implement the metadynamics repulsive potential as a discrete
    /// landscape in m_z covering the range -2.0 to 2.0. Only the range -1.0 to
    /// 1.0 is physically meaningful. When we insert a Gaussian we use mirror
    /// boundary conditions with mirrors at -1.0 and 1.0. This avoids the
    /// potential and the spin system getting stuck at the edges of the range.
    /// To calculate the value of the potential between discrete points we use
    /// linear interpolation between the discrete point above and below the
    /// desired value of the collective variable.
    ///
    /// To avoid expensive recalculation of the total magnetisation we keep
    /// track of the value in 'magnetisation_' and adjust the value every time a
    /// Monte Carlo spin is accepted. In case errors in the magnetisation
    /// compound we do a full recalculation of the magnetisation sum every time
    /// we insert a Gaussian.
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
