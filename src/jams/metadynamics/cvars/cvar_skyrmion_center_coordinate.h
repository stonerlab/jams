//
// Created by ioannis charalampidis on 20/12/2021.
//

#ifndef JAMS_SRC_JAMS_METADYNAMICS_CVARS_CVAR_SKYRMION_CENTER_CORDINATE_H_
#define JAMS_SRC_JAMS_METADYNAMICS_CVARS_CVAR_SKYRMION_CENTER_CORDINATE_H_

#include <jams/metadynamics/caching_collective_variable.h>
#include <jams/containers/vec3.h>
#include <vector>

namespace jams {
class CVarSkyrmionCoreCoordinate : public CachingCollectiveVariable {
 public:
  CVarSkyrmionCoreCoordinate() = default;

  explicit CVarSkyrmionCoreCoordinate(const libconfig::Setting &settings);
  std::string name() override;
  double value() override;

  /// Returns the value of the collective variable after a trial
  /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
  double spin_move_trial_value(int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

  double calculate_expensive_value() override;

 private:

  /// Maps the 2D x-y plane onto two cylinder coordinate systems to allow
  /// calculation of the center of mass with periodic boundaries. The remapped
  /// coordinates on the cylinders are stored in cylinder_remapping_x_ and
  /// cylinder_remapping_y_. Because the lattice is fixed this only needs to
  /// be done once on initialisation.
  void space_remapping();
  double skyrmion_center_of_mass_coordinate(); //Changed to double since I want to return on the coordinate
  // Threshold value of s_z for considering a spin as part of the core of
  // the skyrmion for the purposes of calculating the centre.
  double skyrmion_core_threshold_; //
  // Returns true if the z component of spin crosses the given threshold when
  // changing from s_initial to s_final.
  static bool spin_crossed_threshold(const Vec3 &s_initial, const Vec3 &s_final, const double &threshold);

  std::string name_ = "skyrmion_coordinate_";
  bool value_returned_x = true; //by default return x location.
  //if y is requested by config --> value_returned = false
  //this is use in "skyrmion_center_of_mass_coordinate" to return X OR Y
  bool periodic_x_ = true;
  bool periodic_y_ = true;

  std::vector<Vec3> cylinder_remapping_x_;
  std::vector<Vec3> cylinder_remapping_y_;

};
}

#endif //JAMS_SRC_JAMS_METADYNAMICS_CVARS_CVAR_SKYRMION_CENTER_CORDINATE_H_

