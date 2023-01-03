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

// TODO: Add support for boundary conditions

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
    cvar_file_output_ = jams::config_optional<int>(settings, "cvars_output_steps", 10);
    std::cout << "cvar file output steps : "<< cvar_file_output_ << std::endl;
    gaussian_amplitude_ = config_required<double>(settings, "gaussian_amplitude");

  std::string potential_filename = jams::config_optional<std::string>(settings, "potential_file", "");

  num_cvars_ = settings["collective_variables"].getLength();

  // We currently only support 1D or 2D CV spaces. DO NOT proceed if there are
  // more specified.
  if (num_cvars_ > kMaxDimensions || num_cvars_ < 1) {
    throw std::runtime_error(
        "The number of collective variables should be between 1 and " +
        std::to_string(kMaxDimensions));
  }

  cvars_.resize(num_cvars_);
  cvar_names_.resize(num_cvars_);
  cvar_bcs_.resize(num_cvars_);
  lower_cvar_bc_.resize(num_cvars_);
  upper_cvar_bc_.resize(num_cvars_);
  gaussian_width_.resize(num_cvars_);
  cvar_sample_points_.resize(num_cvars_);
  cvar_range_min_.resize(num_cvars_);
  cvar_range_max_.resize(num_cvars_);

  // Preset the num_samples in each dimension to 1. Then if we are only using
  // 1D our potential will be N x 1 (rather than N x 0!).
  std::fill(std::begin(num_samples_), std::end(num_samples_), 1);

  for (auto i = 0; i < num_cvars_; ++i) {
    // Construct the collective variables from the factor and store pointers
    const auto &cvar_settings = settings["collective_variables"][i];
    cvars_[i].reset(CollectiveVariableFactory::create(cvar_settings));
    cvar_names_[i] = cvars_[i]->name();

    // Set the gaussian width for this collective variable
    gaussian_width_[i] = config_required<double>(cvar_settings,
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
    cvar_sample_points_[i] = linear_space(range_min, range_max, range_step);
    cvar_range_max_[i] = range_max;
    cvar_range_min_[i] = range_min;
    num_samples_[i] = cvar_sample_points_[i].size();

    // Set the lower and upper boundary conditions for this collective variable
    //TODO: Once the mirror boundaries are applied need to generalase these if statements
    //TODO: Currently if lower or upper boundaries are passed from confiq, automatically we set the restoringBC and require the upper and lower boundary, need to generalise.


    if (cvar_settings.exists("lower_restoring_bc_threshold")) {
        //add an if statement to check the name of the boundary condition:
        //if its "restoringBC" do the following. Else jams::unimplemented_error("lower_boundary");
        lower_cvar_bc_[i] = PotentialBCs::RestoringBC;
        lower_restoringBC_threshold_ = jams::config_required<double>(cvar_settings, "lower_restoring_bc_threshold");
        std::cout<< "Lower Restoring Boundary Condition, threshold: "<< lower_restoringBC_threshold_ << std::endl;
        restoringBC_string_constant_ = jams::config_required<double>(cvar_settings, "restoring_bc_spring_constant");
        std::cout<< "Restoring Boundary Condition spring constant: "<< restoringBC_string_constant_ << std::endl;
    }
    else {
      lower_cvar_bc_[i] = PotentialBCs::HardBC;
    }

    if (cvar_settings.exists("upper_restoring_bc_threshold")) {
        //add an if statement to check the name of the boundary condition:
        //if its "restoringBC" do the following. Else jams::unimplemented_error("lower_boundary");
        upper_cvar_bc_[i] = PotentialBCs::RestoringBC;
        upper_restoringBC_threshold_ = jams::config_required<double>(cvar_settings,"upper_restoring_bc_threshold");
        std::cout << "Upper Restoring Boundary Condition, threshold:" <<upper_restoringBC_threshold_ << std::endl;
      restoringBC_string_constant_ = jams::config_required<double>(cvar_settings, "restoring_bc_spring_constant");
      std::cout<< "Restoring Boundary Condition spring constant: "<< restoringBC_string_constant_ << std::endl;

    } else {
      upper_cvar_bc_[i] = PotentialBCs::HardBC;
    }
  }

    potential_.resize(num_samples_[0], num_samples_[1]);

    if (!potential_filename.empty()) {
        std::cout << "Reading potential landscape data from " << potential_filename << "\n" << "Ensure you input the final h5 file from the previous simmulation" <<"\n";
        import_potential(potential_filename);
    }

    cvar_file_.open(jams::output::full_path_filename("metad_cvars.tsv"));
    cvar_file_ << "time";
    for (auto i = 0; i < num_cvars_; ++i) {
      cvar_file_ << " " << cvars_[i]->name();
    }
    cvar_file_ << std::endl;
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

  for (auto n = 0; n < num_cvars_; ++n) {
    cvar_initial[n] = cvars_[n]->value();
    cvar_trial[n] = cvars_[n]->spin_move_trial_value(i, spin_initial, spin_final);
  }

  for (auto n = 0; n < num_cvars_; ++n) {
    if (cvar_bcs_[i] == PotentialBCs::HardBC) {
      if (cvar_initial[n] < cvar_sample_points_[n].front() ||
          cvar_initial[n] > cvar_sample_points_[n].back()) {
        return -kHardBCsPotential;
      }
      if (cvar_trial[n] < cvar_sample_points_[n].front() ||
          cvar_trial[n] > cvar_sample_points_[n].back()) {
        return kHardBCsPotential;
      }
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

  for (auto n = 0; n < num_cvars_; ++n) {
    if (lower_cvar_bc_[n] == PotentialBCs::RestoringBC && cvar_coordinates[n] <= lower_restoringBC_threshold_) {
      return restoringBC_string_constant_ * pow2(cvar_coordinates[n] - lower_restoringBC_threshold_) + potential_(0,0);
    }

    if (upper_cvar_bc_[n] == PotentialBCs::RestoringBC && cvar_coordinates[n] >= upper_restoringBC_threshold_) {
      return restoringBC_string_constant_ * pow2(cvar_coordinates[n] - upper_restoringBC_threshold_) + potential_(num_samples_[n] - 1,0);
    }


	  if (cvar_coordinates[n] < cvar_sample_points_[n].front()
		  || cvar_coordinates[n] > cvar_sample_points_[n].back()) {
		bcs_potential = kHardBCsPotential;
	  }
  }

  for (auto n = 0; n < num_cvars_; ++n) {
    auto lower = std::lower_bound(
        cvar_sample_points_[n].begin(),
        cvar_sample_points_[n].end(),
        cvar_coordinates[n]);

    auto lower_index = std::distance(cvar_sample_points_[n].begin(), lower-1);

    index_lower[n] = lower_index;
    sample_lower[n] = cvar_sample_points_[n][lower_index];
  }

  // TODO: generalise to at least 3D
  assert(num_cvars_ <= kMaxDimensions);


  if (num_cvars_ == 1) {
    auto x1_index = index_lower[0];
    auto x2_index = index_lower[0] + 1;

    return bcs_potential + maths::linear_interpolation(
        cvar_coordinates[0],
        cvar_sample_points_[0][x1_index], potential_(x1_index, 0),
        cvar_sample_points_[0][x2_index], potential_(x2_index, 0));
  }

  if (num_cvars_ == 2) {
    //f(x1,y1)=Q(11) , f(x1,y2)=Q(12), f(x2,y1), f(x2,y2)
    int x1_index = index_lower[0];
    int y1_index = index_lower[1];
    int x2_index = x1_index + 1;
    int y2_index = y1_index + 1;

    double Q11 = potential_(x1_index, y1_index);
    double Q12 = potential_(x1_index, y2_index);
    double Q21 = potential_(x2_index, y1_index);
    double Q22 = potential_(x2_index, y2_index);


    return bcs_potential + maths::bilinear_interpolation(
        cvar_coordinates[0], cvar_coordinates[1],
        cvar_sample_points_[0][x1_index],
        cvar_sample_points_[1][y1_index],
        cvar_sample_points_[0][x2_index],
        cvar_sample_points_[1][y2_index],
        Q11, Q12, Q21, Q22);
  }

  assert(false);
  return 0.0;
}

void jams::MetadynamicsPotential::insert_gaussian(const double& relative_amplitude) {

  // Calculate gaussians along each 1D axis
  std::vector<std::vector<double>> gaussians;
  for (auto n = 0; n < num_cvars_; ++n) {
    gaussians.emplace_back(std::vector<double>(num_samples_[n]));
    auto center = cvars_[n]->value();

    // If we have restoring boundary conditions and we are outside of the
    // central range then we won't be adding any gaussian density so will
    // just zero out the gaussian array
    if (lower_cvar_bc_[n] == PotentialBCs::RestoringBC && center <= lower_restoringBC_threshold_) {
      std::fill(std::begin(gaussians[n]), std::end(gaussians[n]), 0.0);
      // skip setting the gaussians below
      continue;
    }

    if (upper_cvar_bc_[n] == PotentialBCs::RestoringBC && center >= upper_restoringBC_threshold_) {
      std::fill(std::begin(gaussians[n]), std::end(gaussians[n]), 0.0);
      // skip setting the gaussians below
      continue;
    }

    for (auto i = 0; i < num_samples_[n]; ++i) {
      gaussians[n][i] = gaussian(cvar_sample_points_[n][i], center, 1.0, gaussian_width_[n]);
    }
  }

  // If we only have 1D then we need to just have a single element with '1.0'
  // for the second dimension.
  if (num_cvars_ == 1) {
    gaussians.emplace_back(std::vector<double>(1,1.0));
  }

  for (auto i = 0; i < num_samples_[0]; ++i) {
    for (auto j = 0; j < num_samples_[1]; ++j) {
      potential_(i,j) += relative_amplitude * gaussian_amplitude_ * gaussians[0][i] * gaussians[1][j];
    }
  }
  
  if (globals::solver->iteration() % cvar_file_output_ == 0 ) {
    cvar_file_ << globals::solver->time();
    for (auto n = 0; n < num_cvars_; ++n) {
      cvar_file_ << " " << cvars_[n]->value();
    }
    cvar_file_ << std::endl;
  }
}


double jams::MetadynamicsPotential::current_potential() {
  std::array<double,kMaxDimensions> coordinates;
  for (auto n = 0; n < num_cvars_; ++n) {
    coordinates[n] = cvars_[n]->value();
  }

  return potential(coordinates);
}

void jams::MetadynamicsPotential::output() {
  std::ofstream of(jams::output::full_path_filename("metad_potential.tsv"));

  for (auto n = 0; n < num_cvars_; ++n) {
    of << cvar_names_[n] << " ";
  }

  of << "potential_meV" << "\n";

  // TODO: generalise to at least 3D
  assert(num_cvars_ <= kMaxDimensions);
  if (num_cvars_ == 1) {
    for (auto i = 0; i < num_samples_[0]; ++i) {
      of <<  cvar_sample_points_[0][i] << " " << potential_(i,0) << "\n";
    }
    return;
  } else if (num_cvars_ == 2) {
    for (auto i = 0; i < num_samples_[0]; ++i) {
      for (auto j = 0; j < num_samples_[1]; ++j) {
        of <<  cvar_sample_points_[0][i] << " " << cvar_sample_points_[1][j] << " " << potential_(i,j) << "\n";
      }
    }
    return;
  }
  assert(false); // Should not be reachable if num_cvars_ <= kMaxDimensions
}

void jams::MetadynamicsPotential::import_potential(const std::string &filename) {
    std::vector<double> file_data;
    bool first_line = true;
    if (num_cvars_ == 1 ) {
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
        if (file_data.size() != num_samples_[0]) {
             std::cout << num_samples_[0] << " file_data size:"<< file_data.size() << "\n";
            throw std::runtime_error("The " + filename + " has different dimensions from the potential");
        }

        int copy_iterator = 0;
        for (auto i = 0; i < num_samples_[0]; ++i){

                potential_(i,0) = file_data[copy_iterator];
                copy_iterator++;
        }
        potential_file_passed.close();
    }
// **********************************************************************************************
    if(num_cvars_ == 2) {
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
        if (file_data.size() != num_samples_[0] * num_samples_[1]) {
            throw std::runtime_error("The" + filename + "has different dimensions from the potential");
        }

        int copy_iterator = 0;
        for (auto i = 0; i < num_samples_[0]; ++i) {
            for (auto j = 0; j < num_samples_[1]; ++j) {
                potential_(i, j) = file_data[copy_iterator];
            }
            copy_iterator++;
        }
        potential_file_passed.close();
    }
}

