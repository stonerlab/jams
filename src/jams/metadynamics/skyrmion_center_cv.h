//
// Created by ioannis charalampidis on 26/04/2021.
//

#ifndef JAMS_SRC_JAMS_METADYNAMICS_SKYRMION_CENTER_CV_H_
#define JAMS_SRC_JAMS_METADYNAMICS_SKYRMION_CENTER_CV_H_



#include <jams/metadynamics/collective_variable_potential.h>
#include <jams/interface/config.h>
#include <fstream>
#include "jams/containers/multiarray.h"

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
    // map the 2D x-y plane onto two cylinder coordinate systems to allow
    // calculation of the center of mass with periodic boundaries
    void space_remapping();

    // returns true if the z component of spin crosses the given threshold
    bool spin_crossed_threshold(const Vec3& s_initial, const Vec3& s_final, const double& threshold);

    Vec3 calc_center_of_mass(); // I have removed the treshold for now
  double interpolated_2d_potential( const double& y, const double x);
  double gaussian_2D(const double &x, const double &x0, const double &y, const double &y0, const double amplitude) const;
  void skyrmion_output();



  double gaussian_amplitude_;
  double gaussian_width_;
  double histogram_step_size_;
  double skyrmion_threshold_;

  std::vector<double> sample_points_x_;
  std::vector<double> sample_points_y_;
  std::vector<std::vector<double>> potential_2d_;

  Vec3 cached_initial_center_of_mass_ = {0.0, 0.0, 0.0};
  Vec3 cached_trial_center_of_mass_ = {0.0, 0.0, 0.0};

  std::vector<Vec3> tube_x_, tube_y_;

  std::ofstream potential_landscape;
  std::ofstream skyrmion_outfile;

}
;}

#endif //JAMS_SRC_JAMS_METADYNAMICS_SKYRMION_CENTER_CV_H_