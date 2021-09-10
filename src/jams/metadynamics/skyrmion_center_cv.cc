//
// Created by Ioannis Charalampidis on 26/04/2021.
//

#include "jams/metadynamics/skyrmion_center_cv.h"
#include <jams/core/globals.h>
#include <jams/maths/interpolation.h>
#include "jams/maths/functions.h"
#include <libconfig.h++>
#include <fstream>
#include <jams/interface/config.h>
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/helpers/output.h"
#include "jams/helpers/maths.h"
#include <string>
#include <algorithm>

namespace {

std::vector<double> linspace(double min, double max, double steps) {
  assert(min < max);
  std::vector<double> space;
  double value = min;
  while (value < max + steps) {
    space.push_back(value);
    value += steps;
  }

  return space;
}

}

jams::SkyrmionCenterCV::SkyrmionCenterCV(const libconfig::Setting &settings) {
  // ---------------------------------------------------------------------------
  // Read settings
  // ---------------------------------------------------------------------------

  // maximum amplitude of inserted Gaussians in Joules
  // (this can be reduced by tempering in the metadynamics solver)
  gaussian_amplitude_ = jams::config_required<double>(settings, "gaussian_amplitude") / kJoule2meV;

  // width of the gaussian in units of ??
  gaussian_width_ = jams::config_required<double>(settings, "gaussian_width");
  // discretisation width of the metadynamics potential landscape in units of ??
  histogram_step_size_ = jams::config_required<double>(settings, "histogram_step_size");

  std::string potential_filename = jams::config_optional<std::string>(settings, "potential_file", "");

  // ---------------------------------------------------------------------------
  // validate settings
  // ---------------------------------------------------------------------------

  // If histogram_step_size does not divide evenly into the range -1 -> 1 then
  // we will be missing either the start of the end point of the physical range.
  if (!approximately_zero(std::remainder(2.0, histogram_step_size_), 1e-5)) {
    throw std::runtime_error("Invalid value of histogram_step_size: "
                             "histogram_step_size must divide into 2.0 with no remainder");
  }

  // In general our 2D plane is a parallelogram and our x,y minimum and maximum
  // need to be found by trying the coordinates of the different corners. We
  // don't know in advance which points will be used because we could have
  // (for example) this system:
  //
  //    top left ____________ top right
  //             \           \
  //              \           \
  //   bottom left \___________\ bottom right
  //
  // or this system:
  //
  //          top left ____________ top right
  //                 /           /
  //               /           /
  // bottom left /___________/ bottom right
  //
  // So we first find the coordinates of the corners and then find the min and
  // max coordinates. This then allows us to create a rectangular sample space
  //
  //       ___________________
  //      |     /           /|
  //      |   /           /  |
  //      |/___________/_____|
  //
  // The system can never actually reach the triangle in the edges but the
  // metadynamics potential can 'leak' over the edge. This may actually be
  // beneficial in periodic systems. In general we may need to think about how
  // to deal with periodic systems (or periodic on one direction).
  //
  // In practical terms it means the output function will give a rectangle
  // but only the data within parallelogram is part of the surface we sampled.
  //

  periodic_x_ = lattice->is_periodic(0);
  periodic_y_ = lattice->is_periodic(1);

  auto bottom_left  = lattice->get_unitcell().matrix() * Vec3{0.0, 0.0, 0.0};
  auto bottom_right = lattice->get_unitcell().matrix() * Vec3{double(lattice->size(0)), 0.0, 0.0};
  auto top_left     = lattice->get_unitcell().matrix() * Vec3{0.0, double(lattice->size(1)), 0.0};
  auto top_right    = lattice->get_unitcell().matrix() * Vec3{double(lattice->size(0)), double(lattice->size(1)), 0.0};

  auto bounds_x = std::minmax({bottom_left[0], bottom_right[0], top_left[0], top_right[0]});
  auto bounds_y = std::minmax({bottom_left[1], bottom_right[1], top_left[1], top_right[1]});

  //Create the 2d_potential landscape with dimension of the lattice points along x and y
  cv_samples_x_ = linspace(bounds_x.first, bounds_x.second,
                           histogram_step_size_);
  cv_samples_y_ = linspace(bounds_y.first, bounds_y.second,
                           histogram_step_size_); // TODO : dont know why rmax()[1] goes only up to 55.5 that's why I use rmax()[0] for y
  potential_.resize(cv_samples_x_.size(), cv_samples_y_.size());
  zero(potential_);

  if (!potential_filename.empty()) {
    std::cout << "reading potential landscape data from " << potential_filename << "\n";
    import_potential(potential_filename);
  }

  skyrmion_outfile.open(jams::output::full_path_filename("sky_test.tsv"));
  skyrmion_com.open(jams::output::full_path_filename("com_track.tsv"));
  skyrmion_outfile <<"Iteration"<< "	"<< "x" << "	"<< "y" << "	"<< "z" << "\n" ;
  skyrmion_com <<"Iteration"<< "	"<< "x" << "	"<< "y" << "	"<< "z" << "\n" ;
  skyrmion_core_threshold_ = -0.5;

  space_remapping();
  output_remapping();

}

void jams::SkyrmionCenterCV::import_potential(const std::string &filename) {
  std::vector<double> file_data;
  double x_range,y_range,potential_passed;

  std::ifstream potential_file_passed(filename.c_str());

  int line_number = 0;
  for (std::string line; getline(potential_file_passed, line);) {
    if (string_is_comment(line)) {
      continue;
    }
    std::stringstream is(line);
    is >> x_range >> y_range >> potential_passed;

    if (is.bad() || is.fail()) {
      throw std::runtime_error("failed to read line " + std::to_string(line_number));
    }

    file_data.push_back(potential_passed);

    line_number++;
  }

  // If the file data is not the same size as our arrays in the class
  // all sorts of things could go wrong. Stop the simulation here to avoid
  // unintended consequences.
  if (file_data.size() != cv_samples_x_.size() * cv_samples_y_.size()) {
    throw std::runtime_error("potential in file'" + filename + "' is not the same size as in this simulation");
  }

  int copy_iterator = 0;
  for (auto i = 0; i < cv_samples_x_.size(); ++i){
    for(auto j = 0; j < cv_samples_y_.size(); ++j){
      potential_(i, j) = file_data[copy_iterator];
      copy_iterator++;
    }
  }
  potential_file_passed.close();
}


void jams::SkyrmionCenterCV::output() {
  skyrmion_com << solver->iteration()<< "	" <<cached_initial_center_of_mass_[0] << "	" << cached_initial_center_of_mass_[1] << "	" << cached_initial_center_of_mass_[2] << "\n";

  if (solver->iteration()%1000 == 0){
    potential_landscape.open(jams::output::full_path_filename("skyrmion_potential.tsv"));
	for (auto i = 0; i < cv_samples_x_.size(); ++i) {
	  for (auto j = 0; j < cv_samples_y_.size(); ++j) {
		potential_landscape << cv_samples_x_[i] << "	" << cv_samples_y_[j] << "	"
                        << potential_(i,j) << "\n";
	  }
	}
	potential_landscape.close();
  }

  }


void jams::SkyrmionCenterCV::insert_gaussian(const double &relative_amplitude) {
  if (do_first_cache_) {
    cached_initial_center_of_mass_ = skyrmion_center_of_mass();
    cached_trial_center_of_mass_ = cached_initial_center_of_mass_;
    do_first_cache_ = false;
  }
  for (auto i = 0; i < cv_samples_x_.size(); ++i) {
    double g1 = jams::maths::gaussian(cached_initial_center_of_mass_[0], cv_samples_x_[i], gaussian_width_, 1.0);
	  for (auto j = 0; j < cv_samples_y_.size(); ++j) {
      double g2 = jams::maths::gaussian(cached_initial_center_of_mass_[1], cv_samples_y_[j], gaussian_width_, 1.0);
      potential_(i,j) += relative_amplitude * g1 * g2;
	  }
  }
}


double jams::SkyrmionCenterCV::current_potential() {
  if (do_first_cache_) {
    cached_initial_center_of_mass_ = skyrmion_center_of_mass();
    cached_trial_center_of_mass_ = cached_initial_center_of_mass_;
    do_first_cache_ = false;
  }
 return interpolated_potential(cached_initial_center_of_mass_[0],
                               cached_initial_center_of_mass_[1]);
}

double jams::SkyrmionCenterCV::potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final){
  if (do_first_cache_) {
    cached_initial_center_of_mass_ = skyrmion_center_of_mass();
    cached_trial_center_of_mass_ = cached_initial_center_of_mass_;
    do_first_cache_ = false;
  }

  double initial_potential = current_potential(); // save it in a gloabal variable
  double trial_potential;

  Vec3 trial_center_of_mass = cached_initial_center_of_mass_;

  if (spin_crossed_threshold(spin_initial, spin_final, skyrmion_core_threshold_)) {
    trial_center_of_mass = skyrmion_center_of_mass();
    cached_trial_center_of_mass_ = trial_center_of_mass;
  }

  trial_potential = interpolated_potential(trial_center_of_mass[0],
                                           trial_center_of_mass[1]);
  return trial_potential - initial_potential;
}
void jams::SkyrmionCenterCV::spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
  if (do_first_cache_) {
    cached_initial_center_of_mass_ = skyrmion_center_of_mass();
    cached_trial_center_of_mass_ = cached_initial_center_of_mass_;
    do_first_cache_ = false;
  }
  
  // we don't need to check if the threshold was crossed here because
  // cached_trial_center_of_mass_ contains the correct center of mass
  // from the potential_difference() function
  cached_initial_center_of_mass_ = cached_trial_center_of_mass_;
}


Vec3 jams::SkyrmionCenterCV::skyrmion_center_of_mass() {
  using namespace globals;
  using namespace std;

  Mat3 W = lattice->get_unitcell().matrix();
  W[0][2] = 0.0; W[1][2] = 0.0; W[2][2] = 1.0;

  Vec3 tube_center_of_mass_x = {0.0, 0.0, 0.0};
  Vec3 tube_center_of_mass_y = {0.0, 0.0, 0.0};

  Vec3 center_of_mass = {0.0, 0.0, 0.0};

  int num_core_spins = 0;
  for (auto i = 0; i < num_spins; ++i) {
    if (globals::s(i,2) < skyrmion_core_threshold_) {
      tube_center_of_mass_x += cylinder_remapping_x_[i];
      tube_center_of_mass_y += cylinder_remapping_y_[i];
      num_core_spins++;
    }
  }


  if (periodic_x_) {
    double theta_x = atan2(-tube_center_of_mass_x[2], -tube_center_of_mass_x[1]) + kPi;
    center_of_mass[0] = theta_x*lattice->size(0)/(kTwoPi);
  } else {
    center_of_mass[0] = tube_center_of_mass_x[0] / double(num_core_spins);
  }

  if (periodic_y_) {
    double theta_y = atan2(-tube_center_of_mass_y[2], -tube_center_of_mass_y[1]) + kPi;
    center_of_mass[1] = theta_y*lattice->size(1)/(kTwoPi);
  } else {
    center_of_mass[1] = tube_center_of_mass_y[1] / double(num_core_spins);
  }

	return W*center_of_mass;
}


double jams::SkyrmionCenterCV::interpolated_potential(const double &x, const double& y) {
  int x1_index = floor(abs((x - cv_samples_x_[0]) / (cv_samples_x_[1]
                                                    - cv_samples_x_[0])));
  int y1_index = floor(abs((y - cv_samples_y_[0]) / (cv_samples_y_[1]
                                                       - cv_samples_y_[0])));
  int x2_index = x1_index + 1;
  int y2_index = y1_index + 1;

  double Q11 = potential_(x1_index, y1_index);
  double Q12 = potential_(x1_index, y2_index);
  double Q21 = potential_(x2_index, y1_index);
  double Q22 = potential_(x2_index, y2_index);

  return jams::maths::bilinear_interpolation(x, y,
                                             cv_samples_x_[x1_index],
                                             cv_samples_y_[y1_index],
                                             cv_samples_x_[x2_index],
                                             cv_samples_y_[y2_index],
                                             Q11, Q12, Q21, Q22);
}


void jams::SkyrmionCenterCV::space_remapping() {
  // The remapping is done in direct (fractional) space rather than real space
  // because it allows us to handle non-square lattices.

  // find maximum extent of the system for normalisation

  cylinder_remapping_x_.resize(globals::num_spins);
  cylinder_remapping_y_.resize(globals::num_spins);

  // We need to remove the third dimension from the lattice matrix because
  // the remapping is in 2D only.
  //
  // NOTE: This means that we are assuming lattice vectors a,b are in the
  // x,y plane and c is BOTH out of the plane and orthogonal to a,b. i.e.
  // it must be a vector along z. We do a check here for safety.
  auto c_unit_vec = normalize(lattice->get_unitcell().c());
  assert(approximately_zero(c_unit_vec[0], 1e-8)
      && approximately_zero(c_unit_vec[1], 1e-8)
      && approximately_equal(c_unit_vec[2], 1.0, 1e-8));

  Mat3 W = lattice->get_unitcell().matrix();
  W[0][2] = 0.0; W[1][2] = 0.0; W[2][2] = 1.0;

  // map 2D space into a cylinder with y as the axis
  double x_max = lattice->size(0);
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r = inverse(W) * lattice->atom_position(i);

    if (periodic_x_) {
      auto theta_x = (r[0] / x_max) * (kTwoPi);

      auto x = (x_max / (kTwoPi)) * cos(theta_x);
      auto y = r[1];
      auto z = (x_max / (kTwoPi)) * sin(theta_x);
      cylinder_remapping_x_[i] = Vec3{x, y, z};
    } else {
      cylinder_remapping_x_[i] = r;
    }

  }

  // map 2D space into a cylinder with x as the axis
  auto y_max = lattice->size(1);
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r = inverse(W) * lattice->atom_position(i);

    if (periodic_y_) {
      auto theta_y = (r[1] / y_max) * (kTwoPi);

      auto x = r[0];
      auto y = (y_max / (kTwoPi)) * cos(theta_y);
      auto z = (y_max / (kTwoPi)) * sin(theta_y);

      cylinder_remapping_y_[i] = Vec3{x, y, z};
    } else {
      cylinder_remapping_y_[i] = r;
    }
  }
}

bool jams::SkyrmionCenterCV::spin_crossed_threshold(const Vec3 &s_initial,
                                                    const Vec3 &s_final,
                                                    const double &threshold) {
  return (s_initial[2] <= threshold && s_final[2] > threshold) || (s_initial[2] > threshold && s_final[2] <= threshold);
}

void jams::SkyrmionCenterCV::output_remapping() {
  std::ofstream remap_file_x(jams::output::full_path_filename("sky_map_x.tsv"));

  for (auto i = 0; i < globals::num_spins; ++i) {
    remap_file_x << cylinder_remapping_x_[i][0] << " " << cylinder_remapping_x_[i][1] << " " << cylinder_remapping_x_[i][2] << "\n";
  }

  std::ofstream remap_file_y(jams::output::full_path_filename("sky_map_y.tsv"));

  for (auto i = 0; i < globals::num_spins; ++i) {
    remap_file_y << cylinder_remapping_y_[i][0] << " " << cylinder_remapping_y_[i][1] << " " << cylinder_remapping_y_[i][2] << "\n";
  }
}
