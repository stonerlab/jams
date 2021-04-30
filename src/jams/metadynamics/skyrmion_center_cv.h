//
// Created by ioannis charalampidis on 26/04/2021.
//

#ifndef JAMS_SRC_JAMS_METADYNAMICS_SKYRMION_CENTER_CV_H_
#define JAMS_SRC_JAMS_METADYNAMICS_SKYRMION_CENTER_CV_H_



#include <jams/metadynamics/collective_variable_potential.h>
#include <jams/interface/config.h>
#include "fstream"

namespace jams {
class SkyrmionCenterCV : public CollectiveVariablePotential {
 public:
  SkyrmionCenterCV();

  SkyrmionCenterCV(const libconfig::Setting &settings);

  void insert_gaussian(const double& relative_amplitude) override;

  void output() override;

 double potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

 double current_potential() override;

  void spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

 private:
  void create_center_of_mass_mapping();
  void calc_center_of_mass(std::vector<Vec3> &r_com,std::vector<Vec3 > &tube_x_passed, std::vector<Vec3 > &tube_y_passed); // I have removed the treshold for now
  void tubes_update(const int &spin);
  void trial_center_of_mass (Vec3 trial_spin, int spin_index);
//  double collective_coordinate();
  double interpolated_2d_potential( const double& y, const double x);
  double gaussian_2D(const double &x, const double &x0, const double &y, const double &y0, const double amplitude) const;
  void skyrmion_output();



  double gaussian_amplitude_;
  double gaussian_width_;
  double histogram_step_size_;

  std::vector<double> sample_points_x_;
  std::vector<double> sample_points_y_;
  std::vector<std::vector<double>> potential_2d_;

  std::vector<Vec3 > tube_x, tube_y;
  std::vector<double> type_norms;
  std::vector<double> thresholds;
  std::vector<Vec3 > r_com;
  std::vector<Vec3 > trial_r_com;

  std::ofstream potential_landscape;
  std::ofstream skyrmion_outfile;

}
;}

#endif //JAMS_SRC_JAMS_METADYNAMICS_SKYRMION_CENTER_CV_H_