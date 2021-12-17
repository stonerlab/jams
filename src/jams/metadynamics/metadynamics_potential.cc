// metadynamics_potential.cc                                                          -*-C++-*-

#include "metadynamics_potential.h"
#include <jams/metadynamics/collective_variable_factory.h>
#include <jams/maths/interpolation.h>
#include <jams/helpers/output.h>
#include <fstream>

#include <jams/core/solver.h>
#include <jams/core/globals.h>
#include <jams/cuda/cuda_array_kernels.h>

// TODO: Add support for boundary conditions

namespace {
std::vector<double> linear_space(const double min,const double max,const double step) {
  assert(min < max);
  std::vector<double> space;
  double value = min;
  while (value < max+step) {
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


  gaussian_amplitude_ = config_required<double>(settings, "gaussian_amplitude");

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
  gaussian_width_.resize(num_cvars_);
  cvar_sample_points_.resize(num_cvars_);

  // Preset the num_samples in each dimension to 1. Then if we are only using
  // 1D our potential will be N x 1 (rather than N x 0!).
  std::fill(std::begin(num_samples_), std::end(num_samples_), 1);

  for (auto i = 0; i < num_cvars_; ++i) {
    // Construct the collective variables from the factor and store pointers
    const auto& cvar_settings = settings["collective_variables"][i];
    cvars_[i].reset(CollectiveVariableFactory::create(cvar_settings, ::solver->is_cuda_solver()));
    cvar_names_[i] = cvars_[i]->name();

    // Set the gaussian width for this collective variable
    gaussian_width_[i] = config_required<double>(cvar_settings, "gaussian_width");

    // Set the samples along this collective variable axis
    double range_step = config_required<double>(cvar_settings, "range_step");
    double range_min = config_required<double>(cvar_settings, "range_min");
    double range_max = config_required<double>(cvar_settings, "range_max");
    cvar_sample_points_[i] = linear_space(range_min, range_max, range_step);
    num_samples_[i] = cvar_sample_points_[i].size();

    // Set the boundary conditions for this collective variable
    // TODO: need to implement this!
    if (cvar_settings.exists("bcs")) {
      if (lowercase(cvar_settings["bcs"]) == "hard"){
        cvar_bcs_[i] = PotentialBCs::HardBC;
      } else if (lowercase(cvar_settings["bcs"]) == "mirror") {
        cvar_bcs_[i] = PotentialBCs::MirrorBC;
      } else {
        throw std::runtime_error("unknown metadynamics boundary condition");
      }
    } else {
      cvar_bcs_[i] = PotentialBCs::HardBC;
    }
  }

  // TODO: need to fix bug for resizing with std::array to make this general for
  // kMaxDimensions
  potential_.resize(num_samples_[0], num_samples_[1]);
  potential_derivative_.resize(num_samples_[0], num_samples_[1]);

  zero(potential_field_.resize(globals::num_spins, 3));

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

  std::array<double,kMaxDimensions> cvar_initial;
  std::array<double,kMaxDimensions> cvar_trial;

  for (auto n = 0; n < num_cvars_; ++n) {
    cvar_initial[n] = cvars_[n]->value();
    cvar_trial[n] = cvars_[n]->spin_move_trial_value(i, spin_initial, spin_final);
  }

  for (auto n = 0; n < num_cvars_; ++n) {
    if (cvar_bcs_[n] == PotentialBCs::HardBC) {
      if (cvar_initial[n] < cvar_sample_points_[n].front()
          || cvar_initial[n] > cvar_sample_points_[n].back()) {
        return -kHardBCsPotential;
      }
      if (cvar_trial[n] < cvar_sample_points_[n].front()
          || cvar_trial[n] > cvar_sample_points_[n].back()) {
        return kHardBCsPotential;
      }
    }
  }

  return potential(cvar_trial) - potential(cvar_initial);
}


double jams::MetadynamicsPotential::potential(std::array<double,kMaxDimensions> cvar_coordinates) {


  double bcs_potential = 0.0;

  // Apply any hard boundary conditions where the potential is set very large
  // if we are outside of the collective variable's range.
  for (auto n = 0; n < num_cvars_; ++n) {
    if (cvar_bcs_[n] == PotentialBCs::HardBC) {
      if (cvar_coordinates[n] < cvar_sample_points_[n].front()
          || cvar_coordinates[n] > cvar_sample_points_[n].back()) {
        bcs_potential = kHardBCsPotential;
      }
    } else if (cvar_bcs_[n] == PotentialBCs::MirrorBC) {
      if (cvar_coordinates[n] < cvar_sample_points_[n].front()) {
        cvar_coordinates[n] = cvar_sample_points_[n].front() - cvar_coordinates[n];
      }
      if (cvar_coordinates[n] > cvar_sample_points_[n].back()) {
        cvar_coordinates[n] = cvar_sample_points_[n].back() - cvar_coordinates[n];
      }
    }
  }

  return bcs_potential + interpolated_sample_value(potential_, cvar_coordinates);
}

double jams::MetadynamicsPotential::potential_derivative(
    std::array<double, kMaxDimensions> cvar_coordinates) {


  double bcs_potential = 0.0;

  // Apply any hard boundary conditions where the potential is set very large
  // if we are outside of the collective variable's range.

  double sign = 1.0;
  for (auto n = 0; n < num_cvars_; ++n) {
    if (cvar_bcs_[n] == PotentialBCs::HardBC) {
      if (cvar_coordinates[n] < cvar_sample_points_[n].front()
          || cvar_coordinates[n] > cvar_sample_points_[n].back()) {
        bcs_potential = kHardBCsPotential;
      }
    } else if (cvar_bcs_[n] == PotentialBCs::MirrorBC) {
      if (cvar_coordinates[n] < cvar_sample_points_[n].front()) {
        cvar_coordinates[n] = cvar_sample_points_[n].front() - cvar_coordinates[n];
        sign *= -1.0;
      }
      if (cvar_coordinates[n] > cvar_sample_points_[n].back()) {
        cvar_coordinates[n] = cvar_sample_points_[n].back() - cvar_coordinates[n];
        sign *= -1.0;
      }
    }
  }

  return bcs_potential + sign * interpolated_sample_value(potential_derivative_, cvar_coordinates);
}

void jams::MetadynamicsPotential::insert_gaussian(const double& relative_amplitude) {

  std::array<double,kMaxDimensions> cvar_coordinates;
  for (auto n = 0; n < num_cvars_; ++n) {
    cvar_coordinates[n] = cvars_[n]->value();
    if (cvar_bcs_[n] == PotentialBCs::MirrorBC) {
      if (cvar_coordinates[n] < cvar_sample_points_[n].front()) {
        cvar_coordinates[n] = cvar_sample_points_[n].front() - cvar_coordinates[n];
      }
      if (cvar_coordinates[n] > cvar_sample_points_[n].back()) {
        cvar_coordinates[n] = cvar_sample_points_[n].back() - cvar_coordinates[n];
      }
    }
  }


  // Calculate CV distances along each 1D axis
  std::vector<std::vector<double>> distances;
  for (auto n = 0; n < num_cvars_; ++n) {
    distances.emplace_back(std::vector<double>(num_samples_[n]));
    auto center = cvar_coordinates[n];
    for (auto i = 0; i < num_samples_[n]; ++i) {
      distances[n][i] = cvar_sample_points_[n][i] - center;
    }
  }

  // Calculate gaussians along each 1D axis
  std::vector<std::vector<double>> gaussians;
  for (auto n = 0; n < num_cvars_; ++n) {
    gaussians.emplace_back(std::vector<double>(num_samples_[n]));
    for (auto i = 0; i < num_samples_[n]; ++i) {
      gaussians[n][i] = gaussian(distances[n][i], 0.0, 1.0, gaussian_width_[n]);
    }
  }

  // TODO: generalise to at least 3D
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

  for (auto i = 0; i < num_samples_[0]; ++i) {
    for (auto j = 0; j < num_samples_[1]; ++j) {
      // TODO: CHECK THIS EQUATION IS CORRECT
      potential_derivative_(i,j) += (1.0/(gaussian_width_[0]*gaussian_width_[0]))*relative_amplitude * gaussian_amplitude_
          * gaussians[0][i] * (distances[0][i]);
    }
  }

  if (cvar_bcs_[0] == PotentialBCs::MirrorBC) {


    { // lower boundary
      // Calculate CV distances along each 1D axis
      std::vector<std::vector<double>> distances;
      distances.emplace_back(std::vector<double>(num_samples_[0]));
      auto center = cvar_sample_points_[0].front() + (cvar_sample_points_[0].front() - cvar_coordinates[0]);
      for (auto i = 0; i < num_samples_[0]; ++i) {
        distances[0][i] = cvar_sample_points_[0][i] - center;
      }

      // Calculate gaussians along each 1D axis
      std::vector<std::vector<double>> gaussians;
      gaussians.emplace_back(std::vector<double>(num_samples_[0]));
      for (auto i = 0; i < num_samples_[0]; ++i) {
        gaussians[0][i] = gaussian(distances[0][i], 0.0, 1.0,
                                   gaussian_width_[0]);
      }

      // TODO: generalise to at least 3D
      // If we only have 1D then we need to just have a single element with '1.0'
      // for the second dimension.
      gaussians.emplace_back(std::vector<double>(1, 1.0));

      for (auto i = 0; i < num_samples_[0]; ++i) {
        for (auto j = 0; j < num_samples_[1]; ++j) {
          potential_(i, j) +=
              relative_amplitude * gaussian_amplitude_ * gaussians[0][i] *
              gaussians[1][j];
        }
      }

      for (auto i = 0; i < num_samples_[0]; ++i) {
        for (auto j = 0; j < num_samples_[1]; ++j) {
          // TODO: CHECK THIS EQUATION IS CORRECT
          potential_derivative_(i, j) +=
              (1.0 / (gaussian_width_[0] * gaussian_width_[0])) *
              relative_amplitude * gaussian_amplitude_
              * gaussians[0][i] * (distances[0][i]);
        }
      }
    }
    { // upper boundary
      // Calculate CV distances along each 1D axis
      std::vector<std::vector<double>> distances;
      distances.emplace_back(std::vector<double>(num_samples_[0]));
      auto center = cvar_sample_points_[0].back() + (cvar_sample_points_[0].back() - cvar_coordinates[0]);
      for (auto i = 0; i < num_samples_[0]; ++i) {
        distances[0][i] = cvar_sample_points_[0][i] - center;
      }

      // Calculate gaussians along each 1D axis
      std::vector<std::vector<double>> gaussians;
      gaussians.emplace_back(std::vector<double>(num_samples_[0]));
      for (auto i = 0; i < num_samples_[0]; ++i) {
        gaussians[0][i] = gaussian(distances[0][i], 0.0, 1.0,
                                   gaussian_width_[0]);
      }

      // TODO: generalise to at least 3D
      // If we only have 1D then we need to just have a single element with '1.0'
      // for the second dimension.
      gaussians.emplace_back(std::vector<double>(1, 1.0));

      for (auto i = 0; i < num_samples_[0]; ++i) {
        for (auto j = 0; j < num_samples_[1]; ++j) {
          potential_(i, j) +=
              relative_amplitude * gaussian_amplitude_ * gaussians[0][i] *
              gaussians[1][j];
        }
      }

      for (auto i = 0; i < num_samples_[0]; ++i) {
        for (auto j = 0; j < num_samples_[1]; ++j) {
          // TODO: CHECK THIS EQUATION IS CORRECT
          potential_derivative_(i, j) +=
              (1.0 / (gaussian_width_[0] * gaussian_width_[0])) *
              relative_amplitude * gaussian_amplitude_
              * gaussians[0][i] * (distances[0][i]);
        }
      }
    }
  }

  cvar_file_ << ::solver->time();
  for (auto n = 0; n < num_cvars_; ++n) {
    cvar_file_ << " " << cvars_[n]->value();
  }
  cvar_file_ << std::endl;
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


const jams::MultiArray<double, 2>&
jams::MetadynamicsPotential::current_fields() {
  assert(num_cvars_ == 1); // multidimensional fields cvars not yet implemented
  std::array<double,kMaxDimensions> coordinates;
  for (auto n = 0; n < num_cvars_; ++n) {
    coordinates[n] = cvars_[n]->value();
  }


  double scalar = potential_derivative(coordinates);

  cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, cvars_[0]->derivatives().device_data(), 1, potential_field_.device_data(), 1);
  cublasDscal(jams::instance().cublas_handle(), globals::num_spins3, &scalar, potential_field_.device_data(), 1);

//  std::cout << potential_field_(0,0) << " " << potential_field_(0,1) << " " << potential_field_(0,2) << std::endl;
  return potential_field_;
}


double jams::MetadynamicsPotential::interpolated_sample_value(
    const jams::MultiArray<double, 2> &sample_space,
    const std::array<double, kMaxDimensions> &cvar_coordinates) {
  // Lookup points above and below for linear interpolation. We can use the
  // the fact that the ranges are sorted to do a bisection search.


  std::array<double,kMaxDimensions> sample_lower;
  std::array<int,kMaxDimensions> index_lower;

  for (auto n = 0; n < num_cvars_; ++n) {
    auto lower = std::upper_bound(
        cvar_sample_points_[n].begin(),
        cvar_sample_points_[n].end(),
        cvar_coordinates[n]);

    auto lower_index = std::distance(cvar_sample_points_[n].begin(), lower) - 1;
    assert(lower_index >= 0);
    assert(lower_index < cvar_sample_points_[n].size() - 1);

    index_lower[n] = lower_index;
    sample_lower[n] = cvar_sample_points_[n][lower_index];
  }

  // TODO: generalise to at least 3D
  assert(num_cvars_ <= kMaxDimensions);
  if (num_cvars_ == 1) {
    auto x1_index = index_lower[0];
    auto x2_index = index_lower[0] + 1;

    return maths::linear_interpolation(
        cvar_coordinates[0],
        cvar_sample_points_[0][x1_index], sample_space(x1_index, 0),
        cvar_sample_points_[0][x2_index], sample_space(x2_index, 0));
  }

  if (num_cvars_ == 2) {
    //f(x1,y1)=Q(11) , f(x1,y2)=Q(12), f(x2,y1), f(x2,y2)
    int x1_index = index_lower[0];
    int y1_index = index_lower[1];
    int x2_index = x1_index + 1;
    int y2_index = y1_index + 1;

    double Q11 = sample_space(x1_index, y1_index);
    double Q12 = sample_space(x1_index, y2_index);
    double Q21 = sample_space(x2_index, y1_index);
    double Q22 = sample_space(x2_index, y2_index);


    return maths::bilinear_interpolation(
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
