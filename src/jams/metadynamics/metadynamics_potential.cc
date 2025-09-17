// metadynamics_potential.cc                                                          -*-C++-*-

#include "metadynamics_potential.h"
#include "jams/interface/highfive.h"
#include <jams/helpers/exception.h>
#include <jams/metadynamics/collective_variable_factory.h>
#include <jams/maths/interpolation.h>
#include <jams/helpers/output.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <algorithm>
#include <sstream>

#include <sys/stat.h>  // For POSIX stat()
#include <fcntl.h>     // For O_CREAT, O_EXCL
#include <unistd.h>    // For close()

#include <jams/core/solver.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

namespace jams {
template<>
inline MetadynamicsPotential::PotentialBCs
config_required(const libconfig::Setting &setting, const std::string &name) {
  auto format = jams::config_required<std::string>(setting, name);
  if (lowercase(format) == "mirror") {
    return MetadynamicsPotential::PotentialBCs::MirrorBC;
  } else if (lowercase(format) == "hard") {
    return MetadynamicsPotential::PotentialBCs::HardBC;
  } else if (lowercase(format) == "restoring") {
    return MetadynamicsPotential::PotentialBCs::RestoringBC;
  } else {
    throw std::runtime_error("Unknown metadynamics boundary condition value '" + format + "' for setting '" + name + "'");
  }
}
}

namespace {
std::vector<double> linear_space(const double min,const double max,const double step) {
  if (!(min < max)) {
    throw std::runtime_error("linear_space: min must be < max");
  }

  if (!(step > 0.0)) {
    throw std::runtime_error("linear_space: step must be > 0");
  }

  assert(min < max);

  std::vector<double> space;
  double value = min;
  while (less_than_approx_equal(value,max, 1e-4)) {
    space.push_back(value);
    value += step;
  }

  return space;
}
}
// ---------------------------------------------------------------------------
// config settings
// ---------------------------------------------------------------------------
//
// Settings in solver:
//
// gaussian_amplitude : (float) Maximum amplitude of inserted gaussians in meV.
//                     This can be reduced by tempering in the metadynamics
//                     solver.
//
// collective_variables : (list of groups) Each group contains the settings
//                        to be passed to the collective variable constructor.
//
//
// Settings in collective_variables:
//
// name : (string) Name of the collective variable module to use
//
// gaussian_width : (float) Width of the gaussian in the collective coordinate
//                  axis.
//
// range_min : (float) Minimum value for this collective variable axis.
//
// range_max : (float) Maximum value for this collective variable axis.
//
// range_step : (float) Step size for sampling the collective variable along
//              this axis.
//
// Example
// -------
//
//  solver : {
//    module = "monte-carlo-metadynamics-cpu";
//    max_steps  = 100000;
//    gaussian_amplitude = 10.0;
//    gaussian_deposition_stride = 100;
//    output_steps = 100;
//
//    collective_variables = (
//      {
//        name = "topological_charge";
//        gaussian_width = 0.1;
//        range_min = -1.05;
//        range_max = 0.05;
//        range_step = 0.01;
//      },
//      {
//        name = "magnetisation";
//        component = "z";
//        gaussian_width = 0.01;
//        range_min = 0.0;
//        range_max = 1.0;
//        range_step = 0.001;
//      }
//    );
//  };
//

jams::MetadynamicsPotential::MetadynamicsPotential(
    const libconfig::Setting &settings) {
  cvar_output_stride_ = jams::config_optional<int>(settings, "cvars_output_steps", 10);
    std::cout << "cvar file output steps : " << cvar_output_stride_ << std::endl;
  metad_gaussian_amplitude_ = config_required<double>(settings, "gaussian_amplitude");

  std::string potential_filename = jams::config_optional<std::string>(settings, "potential_file", "");

  int num_cvars = settings["collective_variables"].getLength();

  // We currently only support 1D or 2D CV spaces. DO NOT proceed if there are
  // more specified.
  if (num_cvars > kMaxDimensions || num_cvars < 1) {
    throw std::runtime_error(
        "The number of collective variables should be between 1 and " +
        std::to_string(kMaxDimensions));
  }

  cvars_.resize(num_cvars);
  cvar_names_.resize(num_cvars);
  cvar_lower_bcs_.resize(num_cvars);
  cvar_upper_bcs_.resize(num_cvars);
  cvar_gaussian_widths_.resize(num_cvars);
  cvar_sample_coordinates_.resize(num_cvars);
  cvar_range_min_.resize(num_cvars);
  cvar_range_max_.resize(num_cvars);

  // Preset the num_samples in each dimension to 1. Then if we are only using
  // 1D our potential will be N x 1 (rather than N x 0!).
  std::fill(std::begin(num_cvar_sample_coordinates_), std::end(num_cvar_sample_coordinates_), 1);
  std::fill(restoring_bc_upper_threshold_.begin(), restoring_bc_upper_threshold_.end(), 0.0);
  std::fill(restoring_bc_lower_threshold_.begin(), restoring_bc_lower_threshold_.end(), 0.0);
  std::fill(restoring_bc_spring_constant_.begin(), restoring_bc_spring_constant_.end(), 0.0);

  for (auto i = 0; i < num_cvars; ++i) {
    // Construct the collective variables from the factor and store pointers
    const auto &cvar_settings = settings["collective_variables"][i];
    cvars_[i].reset(CollectiveVariableFactory::create(cvar_settings));
    cvar_names_[i] = cvars_[i]->name();

    // Set the gaussian width for this collective variable
    cvar_gaussian_widths_[i] = config_required<double>(cvar_settings, "gaussian_width");
    if (!(std::isfinite(cvar_gaussian_widths_[i]) && cvar_gaussian_widths_[i] > 0.0)) {
      throw std::runtime_error("gaussian_width must be finite and > 0");
    }

    // Set the samples along this collective variable axis

    double range_step = config_required<double>(cvar_settings, "range_step");
    double range_min = config_required<double>(cvar_settings, "range_min");
    double range_max = config_required<double>(cvar_settings, "range_max");
    if (cvar_names_[i] == ("skyrmion_coordinate_x") ||
        cvar_names_[i] == ("skyrmion_coordinate_y")) {
      auto bottom_left = globals::lattice->get_unitcell().matrix() * Vec3{0.0, 0.0, 0.0};
      auto bottom_right = globals::lattice->get_unitcell().matrix() *
                          Vec3{double(globals::lattice->size(0)), 0.0, 0.0};
      auto top_left = globals::lattice->get_unitcell().matrix() *
                      Vec3{0.0, double(globals::lattice->size(1)), 0.0};
      auto top_right = globals::lattice->get_unitcell().matrix() *
                       Vec3{double(globals::lattice->size(0)), double(
                           globals::lattice->size(1)),
                            0.0};
      if (cvar_names_[i] == ("skyrmion_coordinate_x")) {
        auto bounds_x = std::minmax(
            {bottom_left[0], bottom_right[0], top_left[0], top_right[0]});
        range_min = bounds_x.first;
        range_max = bounds_x.second;
      }
      if (cvar_names_[i] == ("skyrmion_coordinate_y")) {
        auto bounds_y = std::minmax(
            {bottom_left[1], bottom_right[1], top_left[1], top_right[1]});
        range_min = bounds_y.first;
        range_max = bounds_y.second;
      }
    }
    cvar_sample_coordinates_[i] = linear_space(range_min, range_max, range_step);
    cvar_range_max_[i] = range_max;
    cvar_range_min_[i] = range_min;
    num_cvar_sample_coordinates_[i] = cvar_sample_coordinates_[i].size();

    // Set the lower and upper boundary conditions for this collective variable
    //TODO: Once the mirror boundaries are applied need to generalase these if statements
    //TODO: Currently if lower or upper boundaries are passed from confiq, automatically we set the restoringBC and require the upper and lower boundary, need to generalise.


    // Set 'HardBC' as the default if no boundary condition is given in the settings
    cvar_upper_bcs_[i] = MetadynamicsPotential::PotentialBCs::HardBC;

    // Read the boundary condition type from the settings. Reading of
    // 'upper_restoring_bc_threshold' is for backwards compatibility with
    // earlier versions of the code before the 'upper_boundary_condition'
    // setting was implemented and 'upper_restoring_bc_threshold' was the
    // key setting to turn on restoring RestoringBC.
    if (cvar_settings.exists("upper_bc")) {
      cvar_upper_bcs_[i] =
          jams::config_required<PotentialBCs>(
              cvar_settings, "upper_bc");
    } else if (cvar_settings.exists("upper_restoring_bc_threshold")) {
      cvar_upper_bcs_[i] = PotentialBCs::RestoringBC;
    }

    // The same as above but for the lower boundary conditions
    cvar_lower_bcs_[i] = PotentialBCs::HardBC;

    if (cvar_settings.exists("lower_bc")) {
      cvar_lower_bcs_[i] =
          jams::config_required<PotentialBCs>(
              cvar_settings, "lower_bc");
    } else if (cvar_settings.exists("lower_restoring_bc_threshold")) {
      cvar_lower_bcs_[i] = PotentialBCs::RestoringBC;
    }

    // Read additional settings for the boundary conditions
    if (cvar_upper_bcs_[i] == PotentialBCs::RestoringBC) {
      restoring_bc_upper_threshold_[i] =
          jams::config_required<double>(
              cvar_settings, "upper_restoring_bc_threshold");
      restoring_bc_spring_constant_[i] =
          jams::config_required<double>(
            cvar_settings, "restoring_bc_spring_constant");
    }

    if (cvar_lower_bcs_[i] == PotentialBCs::RestoringBC) {
      restoring_bc_lower_threshold_[i] =
          jams::config_required<double>(
              cvar_settings, "lower_restoring_bc_threshold");
      restoring_bc_spring_constant_[i] =
          jams::config_required<double>(
              cvar_settings, "restoring_bc_spring_constant");
    }
  }

    zero(metad_potential_.resize(num_cvar_sample_coordinates_[0], num_cvar_sample_coordinates_[1]));
    zero(metad_potential_delta_.resize(num_cvar_sample_coordinates_[0], num_cvar_sample_coordinates_[1]));

    if (!potential_filename.empty()) {
        std::cout << "Reading potential landscape data from " << potential_filename << "\n" << "Ensure you input the final h5 file from the previous simulation" <<"\n";
        import_potential(potential_filename);
    }

    cvar_output_file_.open(jams::output::full_path_filename("metad_cvars.tsv"));
    cvar_output_file_ << "time";
    for (auto i = 0; i < cvars_.size(); ++i) {
      cvar_output_file_ << " " << cvars_[i]->name();
    }
    cvar_output_file_ << std::endl;
}


void jams::MetadynamicsPotential::spin_update(int i, const Vec3 &spin_initial,
                                              const Vec3 &spin_final) {
  // Signal to the CollectiveVariables that they should do any internal work
  // needed due to a spin being accepted (usually related to caching).
  for (const auto& cvar : cvars_) {
      cvar->spin_move_accepted(i, spin_initial, spin_final);
  	}
}


double jams::MetadynamicsPotential::potential_difference(
    int i, const Vec3 &spin_initial, const Vec3 &spin_final) {

  std::array<double,kMaxDimensions> cvar_initial = {0.0, 0.0};
  std::array<double,kMaxDimensions> cvar_trial = {0.0, 0.0};

  for (auto n = 0; n < cvars_.size(); ++n) {
    cvar_initial[n] = cvars_[n]->value();
    cvar_trial[n] = cvars_[n]->spin_move_trial_value(i, spin_initial, spin_final);
  }

  for (auto n = 0; n < cvars_.size(); ++n) {
    if (cvar_lower_bcs_[n] == PotentialBCs::HardBC && cvar_initial[n] < cvar_sample_coordinates_[n].front()) {
      return -kHardBCsPotential;
    }
    if (cvar_upper_bcs_[n] == PotentialBCs::HardBC && cvar_initial[n] > cvar_sample_coordinates_[n].back()) {
      return -kHardBCsPotential;
    }
    if (cvar_lower_bcs_[n] == PotentialBCs::HardBC && cvar_trial[n] < cvar_sample_coordinates_[n].front()) {
      return kHardBCsPotential;
    }
    if (cvar_upper_bcs_[n] == PotentialBCs::HardBC && cvar_trial[n] > cvar_sample_coordinates_[n].back()) {
      return kHardBCsPotential;
    }
  }
  return potential(cvar_trial) - potential(cvar_initial);
}


double jams::MetadynamicsPotential::potential(const std::array<double,kMaxDimensions>& cvar_coordinates) {
  assert(cvar_coordinates.size() > 0 && cvar_coordinates.size() <= kMaxDimensions);

  // We must use cvar_coordinates within the potential function and never
  // cvars_[n]->value(). The later will only every return the current value of
  // the cvar whereas potential() will be called with trial coordinates also.


  // Lookup points above and below for linear interpolation. We can use the
  // the fact that the ranges are sorted to do a bisection search.


  double bcs_potential = 0.0;

  // Apply restoring boundary conditions
  for (auto n = 0; n < cvars_.size(); ++n) {
    if (cvar_lower_bcs_[n] == PotentialBCs::RestoringBC && cvar_coordinates[n] <= restoring_bc_lower_threshold_[n]) {
      return interpolated_potential(cvar_coordinates) + restoring_bc_spring_constant_[n] * pow2(cvar_coordinates[n] - restoring_bc_lower_threshold_[n]);
    }
    if (cvar_upper_bcs_[n] == PotentialBCs::RestoringBC && cvar_coordinates[n] >= restoring_bc_upper_threshold_[n]) {
      return interpolated_potential(cvar_coordinates) + restoring_bc_spring_constant_[n] * pow2(cvar_coordinates[n] - restoring_bc_upper_threshold_[n]);
    }
  }

  for (auto n = 0; n < cvars_.size(); ++n) {
	  if (cvar_coordinates[n] < cvar_sample_coordinates_[n].front()
		  || cvar_coordinates[n] > cvar_sample_coordinates_[n].back()) {
		return kHardBCsPotential;
	  }
  }


  return interpolated_potential(cvar_coordinates);

  assert(false);
}


void jams::MetadynamicsPotential::add_gaussian_to_potential(
    const double relative_amplitude, const std::array<double,kMaxDimensions> center) {

  // Calculate gaussians along each 1D axis
  std::vector<std::vector<double>> gaussians;
  for (auto n = 0; n < cvars_.size(); ++n) {
    gaussians.emplace_back(std::vector<double>(num_cvar_sample_coordinates_[n]));

    for (auto i = 0; i < num_cvar_sample_coordinates_[n]; ++i) {
      gaussians[n][i] = gaussian(cvar_sample_coordinates_[n][i], center[n], 1.0, cvar_gaussian_widths_[n]);
    }
  }

  // If we only have 1D then we need to just have a single element with '1.0'
  // for the second dimension.
  if (cvars_.size() == 1) {
    gaussians.emplace_back(std::vector<double>(1,1.0));
  }

  for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
    for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
      metad_potential_(i, j) += relative_amplitude * metad_gaussian_amplitude_ * gaussians[0][i] * gaussians[1][j];
    }
  }

  for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
    for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
      metad_potential_delta_(i, j) += relative_amplitude * metad_gaussian_amplitude_ * gaussians[0][i] * gaussians[1][j];
    }
  }
}

double jams::MetadynamicsPotential::interpolated_potential(const std::array<double, kMaxDimensions> &cvar_coordinates) {
  const auto lower_indices = potential_grid_indices(cvar_coordinates);

  assert(cvars_.size() <= kMaxDimensions);

  const int i0 = lower_indices[0];
  const int i1 = std::min(i0 + 1, num_cvar_sample_coordinates_[0] - 1);

  if (cvars_.size() == 1) {
    // clamp
    if (i0 == i1) return metad_potential_(i0, 0);

    return maths::linear_interpolation(
        cvar_coordinates[0],
        cvar_sample_coordinates_[0][i0], metad_potential_(i0, 0),
        cvar_sample_coordinates_[0][i1], metad_potential_(i1, 0));
  }

  if (cvars_.size() == 2) {
    const int j0 = lower_indices[1];
    const int j1 = std::min(j0 + 1, num_cvar_sample_coordinates_[1] - 1);

    // clamp
    if (i0 == i1 && j0 == j1) return metad_potential_(i0, j0);

    //f(i0,j0)=Q(11) , f(i0,j1)=Q(12), f(i1,j0), f(i1,j1)
    const double Q00 = metad_potential_(i0, j0);
    const double Q01 = metad_potential_(i0, j1);
    const double Q10 = metad_potential_(i1, j0);
    const double Q11 = metad_potential_(i1, j1);

    return maths::bilinear_interpolation(
        cvar_coordinates[0], cvar_coordinates[1],
        cvar_sample_coordinates_[0][i0], cvar_sample_coordinates_[1][j0],
        cvar_sample_coordinates_[0][i1], cvar_sample_coordinates_[1][j1],
        Q00, Q01, Q10, Q11);
  }

  return 0.0;
}

std::array<int, jams::MetadynamicsPotential::kMaxDimensions> jams::MetadynamicsPotential::potential_grid_indices(
  const std::array<double, kMaxDimensions> &cvar_coordinates) {

  std::array<int,kMaxDimensions> index_lower;

  for (auto n = 0; n < cvars_.size(); ++n) {
    auto lower = std::lower_bound(
        cvar_sample_coordinates_[n].begin(),
        cvar_sample_coordinates_[n].end(),
        cvar_coordinates[n]);

    auto lower_index = std::distance(cvar_sample_coordinates_[n].begin(), lower - 1);

    index_lower[n] = lower_index;
  }

  return index_lower;
}

void jams::MetadynamicsPotential::insert_gaussian(const double& relative_amplitude) {

  std::array<double,kMaxDimensions> center;
  for (auto n = 0; n < cvars_.size(); ++n) {
    center[n] = cvars_[n]->value();
  }
  add_gaussian_to_potential(relative_amplitude, center);

  // This deals with any mirror boundaries by adding virtual Gaussians outside
  // the normal range. Note that these are only treated in a quasi 1D way.
  // For example, if we have a 2D system where all boundaries are mirrored
  // in the following diagram 'x' is the 'real' Gaussian, 'o' are the inserted
  // mirror Gaussians and '.' are the locations where one might insert a
  // gaussian if treating this in a true multidimensional fashion--but which
  // we DON'T insert in the code below.
  //
  //     virtual        |     virtual      |      virtual
  //                    |                  |
  //                    |                  |
  //                 .  |  o               |               .
  // +------------------+------------------+------------------+
  //                 o  |  x               |               o
  //                    |                  |
  //                    |                  |
  //     virtual        | 'real' potential |      virtual
  // +------------------+------------------+------------------+
  //                    |                  |
  //                    |                  |
  //                    |                  |
  //     virtual     .  |  o               |      virtual  .


  for (auto n = 0; n < cvars_.size(); ++n) {
    if (cvar_lower_bcs_[n] == PotentialBCs::MirrorBC) {
      auto virtual_center = center;
      virtual_center[n] = 2*cvar_range_min_[n] - virtual_center[n];
      add_gaussian_to_potential(relative_amplitude, virtual_center);
    }

    if (cvar_upper_bcs_[n] == PotentialBCs::MirrorBC) {
      auto virtual_center = center;
      virtual_center[n] = 2*cvar_range_max_[n] - virtual_center[n];
      add_gaussian_to_potential(relative_amplitude, virtual_center);
    }
  }


  if (globals::solver->iteration() % cvar_output_stride_ == 0 ) {
    cvar_output_file_ << globals::solver->time();
    for (auto n = 0; n < cvars_.size(); ++n) {
      cvar_output_file_ << " " << cvars_[n]->value();
    }
    cvar_output_file_ << std::endl;
  }
}


double jams::MetadynamicsPotential::current_potential() {
  std::array<double,kMaxDimensions> coordinates;
  for (auto n = 0; n < cvars_.size(); ++n) {
    coordinates[n] = cvars_[n]->value();
  }

  return potential(coordinates);
}

void jams::MetadynamicsPotential::output() {
  std::ofstream of(jams::output::full_path_filename("metad_potential.tsv"));

  for (auto n = 0; n < cvars_.size(); ++n) {
    of << cvar_names_[n] << " ";
  }

  of << "potential_meV" << "\n";

  // TODO: generalise to at least 3D
  assert(cvars_.size() <= kMaxDimensions);
  if (cvars_.size() == 1) {
    for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
      of << cvar_sample_coordinates_[0][i] << " " << metad_potential_(i, 0) << "\n";
    }
    return;
  } else if (cvars_.size() == 2) {
    for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
      for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
        of << cvar_sample_coordinates_[0][i] << " " << cvar_sample_coordinates_[1][j] << " " << metad_potential_(i, j) << "\n";
      }
    }
    return;
  }
  assert(false); // Should not be reachable if cvars_.size() <= kMaxDimensions
}

void jams::MetadynamicsPotential::import_potential(const std::string &filename) {
    std::vector<double> file_data;
    bool first_line = true;
    if (cvars_.size() == 1 ) {
        double first_cvar, potential_passed;
        std::ifstream potential_file_passed(filename.c_str());

        int line_number = 0;
        for (std::string line; getline(potential_file_passed, line);) {
            if (string_is_comment(line)) {
                continue;
            }
            //ignore the title
            if (first_line){
                first_line = false;
                continue;
            }

            std::stringstream is(line);
            is >> first_cvar >> potential_passed;

            if (is.bad() || is.fail()) {
                throw std::runtime_error("failed to read line " + std::to_string(line_number));
            }

            file_data.push_back(potential_passed);

            line_number++;
        }
        // If the file data is not the same size as our arrays in the class
        // all sorts of things could go wrong. Stop the simulation here to avoid
        // unintended consequences.
        if (file_data.size() != num_cvar_sample_coordinates_[0]) {
             std::cout << num_cvar_sample_coordinates_[0] << " file_data size:" << file_data.size() << "\n";
            throw std::runtime_error("The " + filename + " has different dimensions from the potential");
        }

        int copy_iterator = 0;
        for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i){

          metad_potential_(i, 0) = file_data[copy_iterator];
                copy_iterator++;
        }
        potential_file_passed.close();
    }
// **********************************************************************************************
    if(cvars_.size() == 2) {
        double first_cvar, second_cvar, potential_passed;
        std::ifstream potential_file_passed(filename.c_str());

        int line_number = 0;
        for (std::string line; getline(potential_file_passed, line);) {
            if (string_is_comment(line)) {
                continue;
            }
            // ignore the title
            if (first_line){
                first_line = false;
                continue;
            }

            std::stringstream is(line);
            is >> first_cvar >> second_cvar >> potential_passed;

            file_data.push_back(potential_passed);

            line_number++;
        }

        // If the file data is not the same size as our arrays in the class
        // all sorts of things could go wrong. Stop the simulation here to avoid
        // unintended consequences.
        if (file_data.size() != num_cvar_sample_coordinates_[0] * num_cvar_sample_coordinates_[1]) {
            throw std::runtime_error("The " + filename + " has different dimensions from the potential");
        }

        int copy_iterator = 0;
        for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
            for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
              metad_potential_(i, j) = file_data[copy_iterator];
            }
            copy_iterator++;
        }
        potential_file_passed.close();
    }
}

void jams::MetadynamicsPotential::synchronise_shared_potential(const std::string &file_name) {
  auto lock_file_name = file_name + ".lock";
  int lock_fd = output::lock_file(lock_file_name);

  MultiArray<double, kMaxDimensions> shared_potential(metad_potential_.shape());
  { // Scoping guards to make sure the hdf5 file is closed before we unlock the lock file
    HighFive::File file(file_name, HighFive::File::ReadWrite | HighFive::File::Create);

    if (!file.exist("shared_potential")) {
      zero(shared_potential);
      file.createDataSet<double>("shared_potential", HighFive::DataSpace::From(shared_potential));
    } else {
      file.getDataSet("shared_potential").read(shared_potential);
    }

    for (auto i = 0; i < metad_potential_.size(0); ++i) {
      for (auto j = 0; j < metad_potential_.size(1); ++j) {
        shared_potential(i, j) += metad_potential_delta_(i, j);
      }
    }

    file.getDataSet("shared_potential").write(shared_potential);

    file.flush();
  }

  output::unlock_file(lock_fd);

  metad_potential_ = shared_potential;
  zero(metad_potential_delta_);
}



