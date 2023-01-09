// metadynamics_potential.cc                                                          -*-C++-*-

#include "metadynamics_potential.h"
#include <jams/helpers/exception.h>
#include <jams/metadynamics/collective_variable_factory.h>
#include <jams/maths/interpolation.h>
#include <jams/helpers/output.h>
#include <fstream>
#include <iostream>

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
    throw std::runtime_error("Unknown metadynamics boundary condition: " + name);
  }
}
}

namespace {
std::vector<double> linear_space(const double min,const double max,const double step) {
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

  for (auto i = 0; i < num_cvars; ++i) {
    // Construct the collective variables from the factor and store pointers
    const auto &cvar_settings = settings["collective_variables"][i];
    cvars_[i].reset(CollectiveVariableFactory::create(cvar_settings));
    cvar_names_[i] = cvars_[i]->name();

    // Set the gaussian width for this collective variable
    cvar_gaussian_widths_[i] = config_required<double>(cvar_settings,
                                                       "gaussian_width");

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
      restoring_bc_upper_threshold_ =
          jams::config_required<double>(
              cvar_settings, "upper_restoring_bc_threshold");
      restoring_bc_spring_constant_ =
          jams::config_required<double>(
            cvar_settings, "restoring_bc_spring_constant");
    }

    if (cvar_lower_bcs_[i] == PotentialBCs::RestoringBC) {
      restoring_bc_lower_threshold_ =
          jams::config_required<double>(
              cvar_settings, "lower_restoring_bc_threshold");
      restoring_bc_spring_constant_ =
          jams::config_required<double>(
              cvar_settings, "restoring_bc_spring_constant");
    }
  }

    zero(metad_potential_.resize(num_cvar_sample_coordinates_[0], num_cvar_sample_coordinates_[1]));

    if (!potential_filename.empty()) {
        std::cout << "Reading potential landscape data from " << potential_filename << "\n" << "Ensure you input the final h5 file from the previous simmulation" <<"\n";
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
    if (cvar_lower_bcs_[i] == PotentialBCs::HardBC && cvar_initial[n] < cvar_sample_coordinates_[n].front()) {
      return -kHardBCsPotential;
    }
    if (cvar_upper_bcs_[i] == PotentialBCs::HardBC && cvar_initial[n] > cvar_sample_coordinates_[n].back()) {
      return -kHardBCsPotential;
    }
    if (cvar_lower_bcs_[i] == PotentialBCs::HardBC && cvar_trial[n] < cvar_sample_coordinates_[n].front()) {
      return kHardBCsPotential;
    }
    if (cvar_upper_bcs_[i] == PotentialBCs::HardBC && cvar_trial[n] > cvar_sample_coordinates_[n].back()) {
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

  std::array<double,kMaxDimensions> sample_lower;
  std::array<int,kMaxDimensions> index_lower;

  double bcs_potential = 0.0;

  for (auto n = 0; n < cvars_.size(); ++n) {
    if (cvar_lower_bcs_[n] == PotentialBCs::RestoringBC && cvar_coordinates[n] <= restoring_bc_lower_threshold_) {
      return restoring_bc_spring_constant_ * pow2(cvar_coordinates[n] - restoring_bc_lower_threshold_) + metad_potential_(0, 0);
    }

    if (cvar_upper_bcs_[n] == PotentialBCs::RestoringBC && cvar_coordinates[n] >= restoring_bc_upper_threshold_) {
      return restoring_bc_spring_constant_ * pow2(cvar_coordinates[n] - restoring_bc_upper_threshold_) + metad_potential_(num_cvar_sample_coordinates_[n] - 1, 0);
    }


	  if (cvar_coordinates[n] < cvar_sample_coordinates_[n].front()
		  || cvar_coordinates[n] > cvar_sample_coordinates_[n].back()) {
		bcs_potential = kHardBCsPotential;
	  }
  }

  for (auto n = 0; n < cvars_.size(); ++n) {
    auto lower = std::lower_bound(
        cvar_sample_coordinates_[n].begin(),
        cvar_sample_coordinates_[n].end(),
        cvar_coordinates[n]);

    auto lower_index = std::distance(cvar_sample_coordinates_[n].begin(), lower - 1);

    index_lower[n] = lower_index;
    sample_lower[n] = cvar_sample_coordinates_[n][lower_index];
  }

  assert(cvars_.size() <= kMaxDimensions);

  if (cvars_.size() == 1) {
    auto x1_index = index_lower[0];
    auto x2_index = index_lower[0] + 1;

    return bcs_potential + maths::linear_interpolation(
        cvar_coordinates[0],
        cvar_sample_coordinates_[0][x1_index], metad_potential_(x1_index, 0),
        cvar_sample_coordinates_[0][x2_index], metad_potential_(x2_index, 0));
  }

  if (cvars_.size() == 2) {
    //f(x1,y1)=Q(11) , f(x1,y2)=Q(12), f(x2,y1), f(x2,y2)
    int x1_index = index_lower[0];
    int y1_index = index_lower[1];
    int x2_index = x1_index + 1;
    int y2_index = y1_index + 1;

    double Q11 = metad_potential_(x1_index, y1_index);
    double Q12 = metad_potential_(x1_index, y2_index);
    double Q21 = metad_potential_(x2_index, y1_index);
    double Q22 = metad_potential_(x2_index, y2_index);


    return bcs_potential + maths::bilinear_interpolation(
        cvar_coordinates[0], cvar_coordinates[1],
        cvar_sample_coordinates_[0][x1_index],
        cvar_sample_coordinates_[1][y1_index],
        cvar_sample_coordinates_[0][x2_index],
        cvar_sample_coordinates_[1][y2_index],
        Q11, Q12, Q21, Q22);
  }

  assert(false);
  return 0.0;
}


void jams::MetadynamicsPotential::add_gaussian_to_potential(
    const double relative_amplitude, const std::array<double,kMaxDimensions> center) {

  // Calculate gaussians along each 1D axis
  std::vector<std::vector<double>> gaussians;
  for (auto n = 0; n < cvars_.size(); ++n) {
    gaussians.emplace_back(std::vector<double>(num_cvar_sample_coordinates_[n]));

    // If we have restoring boundary conditions and we are outside of the
    // central range then we won't be adding any gaussian density so will
    // just zero out the gaussian array
    if (cvar_lower_bcs_[n] == PotentialBCs::RestoringBC && center[n] <= restoring_bc_lower_threshold_) {
      std::fill(std::begin(gaussians[n]), std::end(gaussians[n]), 0.0);
      // skip setting the gaussians below
      continue;
    }

    if (cvar_upper_bcs_[n] == PotentialBCs::RestoringBC && center[n] >= restoring_bc_upper_threshold_) {
      std::fill(std::begin(gaussians[n]), std::end(gaussians[n]), 0.0);
      // skip setting the gaussians below
      continue;
    }

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
      virtual_center[n] = cvar_range_min_[n] - virtual_center[n];
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
            //ingore the title
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
            //ingore the title
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
            throw std::runtime_error("The" + filename + "has different dimensions from the potential");
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

