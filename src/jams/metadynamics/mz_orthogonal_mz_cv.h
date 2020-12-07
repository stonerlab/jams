//
// Created by ioannis charalampidis on 28/11/2020.
//

#ifndef JAMS_METADYNAMICS_MZ_ORTHOGONAL_MZ_CV_H_
#define JAMS_METADYNAMICS_MZ_ORTHOGONAL_MZ_CV_H_

#include <jams/metadynamics/collective_variable_potential.h>
#include <jams/interface/config.h>
#include <fstream>

namespace jams {
class MzOrthogonalMzCV : public CollectiveVariablePotential {
 public:

  MzOrthogonalMzCV();

  MzOrthogonalMzCV(const libconfig::Setting &settings);

  void insert_gaussian(const double& relative_amplitude) override;

  void output() override;

  double potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

  void spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

 private:
  double collective_coordinate();

  double interpolated_2d_potential( const double& m, const double m_p);

  static Vec3 calculate_total_magnetisation();

  double amplitude_tempering (const double &m, const double &m_p);

  static inline double mz_perpendicular(Vec3 &magnetisation);

  double gaussian_2D(const double &x, const double &x0, const double &y, const double &y0, const double amplitude) const;

  double energy_barrier_calculation();

  Vec3 magnetisation_{};
  double mz_perpendicular_{};

   double gaussian_amplitude_;
   double gaussian_width_;
  double tempered_amplitude_;
  double bias_temp_;
  double histogram_step_size_;

  std::vector<double> sample_points_mz_;
  std::vector<double> sample_points_mz_perpendicular_;
  std::vector<std::vector<double>> potential_2d_;

  std::ofstream potential;
  std::ofstream metadynamics_simulation_parameters;




};
}
#endif //JAMS_METADYNAMICS_MZ_ORTHOGONAL_MZ_CV_H_
