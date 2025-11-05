// metadynamics_potential.cc                                                          -*-C++-*-

#include "metadynamics_potential.h"
#include "jams/interface/highfive.h"
#include <jams/helpers/exception.h>
#include <jams/metadynamics/collective_variable_factory.h>
#include <jams/maths/interpolation.h>
#include <jams/helpers/output.h>
#include <jams/helpers/container_utils.h>
#include <cmath>
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

#include "jams/macros.h"

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

  do_metad_potential_interpolation_ = config_optional<bool>(settings, "interpolation", true);
  metad_gaussian_amplitude_ = config_required<double>(settings, "gaussian_amplitude");

  std::string potential_filename = jams::config_optional<std::string>(settings, "potential_file", "");

  int num_cvars = settings["collective_variables"].getLength();

  // We only support a limited, hardcoded number of CV dimensions. DO NOT proceed if there are more CVs specified.
  if (num_cvars > kNumCVars || num_cvars < 1) {
    throw std::runtime_error(
        "The number of collective variables should be between 1 and " +
        std::to_string(kNumCVars));
  }

  cvars_.resize(num_cvars);
  cvar_names_.resize(num_cvars);
  cvar_lower_bcs_.resize(num_cvars);
  cvar_upper_bcs_.resize(num_cvars);
  cvar_gaussian_widths_.resize(num_cvars);
  cvar_sample_coordinates_.resize(num_cvars);
  cvar_range_min_.resize(num_cvars);
  cvar_range_max_.resize(num_cvars);
  cvar_inv_step_.resize(num_cvars);

  zero_all(restoring_bc_upper_threshold_, restoring_bc_lower_threshold_, restoring_bc_spring_constant_);
  zero_all(mirror_bc_upper_threshold_, mirror_bc_lower_threshold_);

  // Preset the num_samples in each dimension to 1. Then if we are only using
  // 1D our potential will be N x 1 (rather than N x 0!).
  std::fill(std::begin(num_cvar_sample_coordinates_), std::end(num_cvar_sample_coordinates_), 1);

  std::fill(std::begin(cvar_lower_bcs_), std::end(cvar_lower_bcs_), MetadynamicsPotential::PotentialBCs::NoBC);
  std::fill(std::begin(cvar_upper_bcs_), std::end(cvar_upper_bcs_), MetadynamicsPotential::PotentialBCs::NoBC);

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
    double range_min = config_required<double>(cvar_settings, "range_min");
    double range_max = config_required<double>(cvar_settings, "range_max");

    // If no step size is given then use width / 5.
    // This is the same as PLUMED default and completely by accident, roughly what we were tending to use.
    double range_step = config_optional<double>(cvar_settings, "range_step", cvar_gaussian_widths_[i] / 5.0);

    if (cvar_names_[i] == ("skyrmion_coordinate_x") || cvar_names_[i] == ("skyrmion_coordinate_y")) {
      const auto cell = globals::lattice->get_unitcell().matrix();
      const auto size_a = double(globals::lattice->size(0));
      const auto size_b = double(globals::lattice->size(1));

      auto bottom_left = cell * Vec3{0.0, 0.0, 0.0};
      auto bottom_right = cell * Vec3{size_a , 0.0, 0.0};
      auto top_left = cell * Vec3{0.0, size_b, 0.0};
      auto top_right = cell * Vec3{size_a , size_b, 0.0};

      if (cvar_names_[i] == ("skyrmion_coordinate_x")) {
        std::tie(range_min, range_max) = std::minmax({bottom_left[0], bottom_right[0], top_left[0], top_right[0]});
      }
      if (cvar_names_[i] == ("skyrmion_coordinate_y")) {
        std::tie(range_min, range_max) = std::minmax({bottom_left[1], bottom_right[1], top_left[1], top_right[1]});
      }
    }

    cvar_sample_coordinates_[i] = linear_space(range_min, range_max, range_step);
    cvar_range_max_[i] = range_max;
    cvar_range_min_[i] = range_min;
    cvar_inv_step_[i] = 1.0 / range_step;
    num_cvar_sample_coordinates_[i] = cvar_sample_coordinates_[i].size();

    // Read the boundary condition type from the settings. Reading of
    // 'upper_restoring_bc_threshold' is for backwards compatibility with
    // earlier versions of the code before the 'upper_boundary_condition'
    // setting was implemented and 'upper_restoring_bc_threshold' was the
    // key setting to turn on restoring RestoringBC.
    if (cvar_settings.exists("upper_bc")) {
      cvar_upper_bcs_[i] = jams::config_required<PotentialBCs>(cvar_settings, "upper_bc");
    } else if (cvar_settings.exists("upper_restoring_bc_threshold")) {
      cvar_upper_bcs_[i] = PotentialBCs::RestoringBC;
    } else if (cvar_settings.exists("upper_mirror_bc_threshold")) {
      cvar_upper_bcs_[i] = PotentialBCs::MirrorBC;
    }

    if (cvar_settings.exists("lower_bc")) {
      cvar_lower_bcs_[i] = jams::config_required<PotentialBCs>(cvar_settings, "lower_bc");
    } else if (cvar_settings.exists("lower_restoring_bc_threshold")) {
      cvar_lower_bcs_[i] = PotentialBCs::RestoringBC;
    } else if (cvar_settings.exists("lower_mirror_bc_threshold")) {
      cvar_lower_bcs_[i] = PotentialBCs::MirrorBC;
    }


    // Read additional settings for the boundary conditions
    if (cvar_upper_bcs_[i] == PotentialBCs::RestoringBC) {
      restoring_bc_upper_threshold_[i] =
          jams::config_required<double>(cvar_settings, "upper_restoring_bc_threshold");
      restoring_bc_spring_constant_[i] =
          jams::config_required<double>(cvar_settings, "restoring_bc_spring_constant");
    }

    if (cvar_lower_bcs_[i] == PotentialBCs::RestoringBC) {
      restoring_bc_lower_threshold_[i] =
          jams::config_required<double>(cvar_settings, "lower_restoring_bc_threshold");
      restoring_bc_spring_constant_[i] =
          jams::config_required<double>(cvar_settings, "restoring_bc_spring_constant");
    }

    if (cvar_upper_bcs_[i] == PotentialBCs::MirrorBC) {
      mirror_bc_upper_threshold_[i] =
          jams::config_required<double>(cvar_settings, "upper_mirror_bc_threshold");
    }

    if (cvar_lower_bcs_[i] == PotentialBCs::MirrorBC) {
      mirror_bc_lower_threshold_[i] =
          jams::config_required<double>(cvar_settings, "lower_mirror_bc_threshold");
    }

    // Upper
    if (!std::isfinite(restoring_bc_upper_threshold_[i]))
      throw std::runtime_error("upper_restoring_bc_threshold must be finite");
    if (!(std::isfinite(restoring_bc_spring_constant_[i]) && restoring_bc_spring_constant_[i] >= 0.0))
      throw std::runtime_error("restoring_bc_spring_constant must be finite and >= 0");

    // Lower
    if (!std::isfinite(restoring_bc_lower_threshold_[i]))
      throw std::runtime_error("lower_restoring_bc_threshold must be finite");
    if (!(std::isfinite(restoring_bc_spring_constant_[i]) && restoring_bc_spring_constant_[i] >= 0.0))
      throw std::runtime_error("restoring_bc_spring_constant must be finite and >= 0");
  }


    zero(metad_potential_.resize(num_cvar_sample_coordinates_[0], num_cvar_sample_coordinates_[1]));
    zero(metad_potential_delta_.resize(num_cvar_sample_coordinates_[0], num_cvar_sample_coordinates_[1]));

    if (!potential_filename.empty()) {
        std::cout << "Reading potential landscape data from " << potential_filename << "\n" << "Ensure you input the final h5 file from the previous simulation" <<"\n";
        import_potential(potential_filename);
    }

    cvar_output_file_.open(jams::output::full_path_filename("metad_cvars.tsv"));
    if (!cvar_output_file_) {
      throw std::runtime_error("Failed to open metad_cvars.tsv for writing");
    }

    cvar_output_file_ << "time";
    for (auto i = 0; i < cvars_.size(); ++i) {
      cvar_output_file_ << " " << cvars_[i]->name();
    }
    cvar_output_file_ << " relative_amplitude" << std::endl;

  std::cout << jams::output::section("init metadynamics potential") << std::endl;

  print_settings();
}


void jams::MetadynamicsPotential::spin_update(int i, const Vec3 &spin_initial,
                                              const Vec3 &spin_final) {
  // Signal to the CollectiveVariables that they should do any internal work
  // needed due to a spin being accepted (usually related to caching).
  for (const auto& cvar : cvars_) {
      cvar->spin_move_accepted(i, spin_initial, spin_final);
  	}
}

// Return the difference in the metadynamics potential energy when changing spin i from the state spin_initial to
// spin_trial.
double jams::MetadynamicsPotential::potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
  std::array<double,kNumCVars> cvar_initial = cvar_coordinates();

  std::array<double,kNumCVars> cvar_trial{};
  for (auto n = 0; n < cvars_.size(); ++n) {
    cvar_trial[n] = cvars_[n]->spin_move_trial_value(i, spin_initial, spin_final);
  }

  return full_potential(cvar_trial) - full_potential(cvar_initial);
}


double jams::MetadynamicsPotential::full_potential(const std::array<double,kNumCVars>& cv) {
  assert(cvars_.size() > 0 && cvars_.size() <= kNumCVars);


  // We must use cv within the potential function and never
  // cvars_[n]->value(). The later will only every return the current value of
  // the cvar whereas potential() will be called with trial coordinates also.

  // Apply any hard boundary conditions. If the CV is over the boundary, return a very large energy penalty.
  for (auto n = 0; n < cvars_.size(); ++n) {
    const auto lo = cvar_sample_coordinates_[n].front();
    const auto hi = cvar_sample_coordinates_[n].back();
    if ((cvar_lower_bcs_[n] == PotentialBCs::HardBC && cv[n] < lo)
      || (cvar_upper_bcs_[n] == PotentialBCs::HardBC && cv[n] > hi)) {
      return kHardBCsPotential; // huge positive penalty
    }
  }

  // The interpolated_potential function has clamping built in, so it will return the
  // clamped value at the edge of the grid if the CV is outside of the grid. Per the
  // header documentation, the base potential is taken at the current coordinates
  // (subject to this grid-edge clamp), and restoring penalties are added separately.
  const auto potential = base_potential(cv);

  // Apply restoring boundary conditions as per header docs:
  // - If a CV is beyond its restoring threshold(s), add a spring penalty
  //   0.5 * k * (x - x_thr)^2.
  // - The base potential is evaluated at the current (possibly clamped-to-grid)
  //   coordinates via interpolated_potential; no special threshold clamp.
  // - Penalties from multiple dimensions accumulate across axes.
  double restoring_penalty = 0.0;
  for (auto n = 0; n < cvars_.size(); ++n) {
      if (cvar_lower_bcs_[n] == PotentialBCs::RestoringBC &&
          cv[n] <= restoring_bc_lower_threshold_[n]) {
          restoring_penalty += 0.5 * restoring_bc_spring_constant_[n]
                             * pow2(cv[n] - restoring_bc_lower_threshold_[n]);
      }
      if (cvar_upper_bcs_[n] == PotentialBCs::RestoringBC &&
          cv[n] >= restoring_bc_upper_threshold_[n]) {
          restoring_penalty += 0.5 * restoring_bc_spring_constant_[n]
                             * pow2(cv[n] - restoring_bc_upper_threshold_[n]);
      }
  }

  return potential + restoring_penalty;
}


void jams::MetadynamicsPotential::add_gaussian_to_landscape(
  const std::array<double,kNumCVars> center,
  MultiArray<double,kNumCVars>& landscape) {

  // Calculate gaussians along each 1D axis
  std::vector<std::vector<double>> gaussians;
  for (auto n = 0; n < cvars_.size(); ++n) {
    gaussians.emplace_back(std::vector<double>(num_cvar_sample_coordinates_[n]));

    for (auto i = 0; i < num_cvar_sample_coordinates_[n]; ++i) {
      if (std::abs(cvar_sample_coordinates_[n][i] - center[n]) > kGaussianExtent * cvar_gaussian_widths_[n]) {
        gaussians[n][i] = 0.0;
      } else {
        gaussians[n][i] = gaussian(cvar_sample_coordinates_[n][i], center[n], 1.0, cvar_gaussian_widths_[n]);
      }
    }
  }

  // If we only have 1D then we need to just have a single element with '1.0'
  // for the second dimension.
  if (cvars_.size() == 1) {
    gaussians.emplace_back(std::vector<double>(1,1.0));
  }

  for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
    for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
      landscape(i, j) += gaussians[0][i] * gaussians[1][j];
    }
  }
}

double jams::MetadynamicsPotential::base_potential(const std::array<double, kNumCVars> &cvar_coordinates) {
  if (do_metad_potential_interpolation_) {
    return get_base_potential_interpolated_value(cvar_coordinates);
  }
  return get_base_potential_nearest_value(cvar_coordinates);
}

double jams::MetadynamicsPotential::get_base_potential_nearest_value(const std::array<double, kNumCVars> &cvar_coordinates) {
  const auto lower_indices = potential_grid_indices(cvar_coordinates);

#ifndef NDEBUG
  for (auto n = 0; n < cvars_.size(); ++n) {
    assert(lower_indices[n] >= 0 &&
           lower_indices[n] < num_cvar_sample_coordinates_[n]);
  }
#endif

  // Choose the nearest grid index in each CV dimension (no interpolation).
  // We start from the clamped lower index returned by potential_grid_indices
  // and, if the next index exists and is closer to the coordinate value,
  // we move up by one.

  // Dimension 0
  int i = lower_indices[0];
  if (cvars_.size() >= 1) {
    const auto &grid0 = cvar_sample_coordinates_[0];
    const double x = cvar_coordinates[0];

    if (i + 1 < num_cvar_sample_coordinates_[0]) {
      const double d0 = std::abs(x - grid0[i]);
      const double d1 = std::abs(x - grid0[i + 1]);
      if (d1 < d0) {
        ++i;
      }
    }
  }

  if (cvars_.size() == 1) {
    // 1D case: nearest point along the single CV axis.
    return metad_potential_(i, 0);
  }

  if (cvars_.size() == 2) {
    // Dimension 1
    int j = lower_indices[1];
    const auto &grid1 = cvar_sample_coordinates_[1];
    const double y = cvar_coordinates[1];

    if (j + 1 < num_cvar_sample_coordinates_[1]) {
      const double d0 = std::abs(y - grid1[j]);
      const double d1 = std::abs(y - grid1[j + 1]);
      if (d1 < d0) {
        ++j;
      }
    }

    // 2D case: nearest point on the 2D grid.
    return metad_potential_(i, j);
  }

  assert(false && "Unreachable code: base_potential only supports up to kNumCVars dimensions");
  UNREACHABLE();
}

double jams::MetadynamicsPotential::get_base_potential_interpolated_value(const std::array<double, kNumCVars> &cvar_coordinates) {
  const auto lower_indices = potential_grid_indices(cvar_coordinates);

#ifndef NDEBUG
  for (auto n = 0; n < cvars_.size(); ++n) {
    assert(lower_indices[n] >= 0 && lower_indices[n] < num_cvar_sample_coordinates_[n]);
  }
#endif

  const auto& grid0 = cvar_sample_coordinates_[0];
  const double x = cvar_coordinates[0];
  const bool clamp_lo_x = x <= grid0.front();
  const bool clamp_hi_x = x >= grid0.back();

  int i0 = lower_indices[0];
  int i1 = (clamp_lo_x || clamp_hi_x) ? i0 : std::min(i0 + 1, num_cvar_sample_coordinates_[0] - 1);

  if (cvars_.size() == 1) {
    // clamp
    if (i0 == i1) return metad_potential_(i0, 0);

    return maths::linear_interpolation(
        x, grid0[i0], metad_potential_(i0, 0),
           grid0[i1], metad_potential_(i1, 0));
  }


  if (cvars_.size() == 2) {
    const auto& grid1 = cvar_sample_coordinates_[1];
    const double y = cvar_coordinates[1];
    const bool clamp_lo_y = y <= grid1.front();
    const bool clamp_hi_y = y >= grid1.back();

    int j0 = lower_indices[1];
    int j1 = (clamp_lo_y || clamp_hi_y) ? j0 : std::min(j0 + 1, num_cvar_sample_coordinates_[1] - 1);

    // full clamp (corner)
    if (i0 == i1 && j0 == j1) return metad_potential_(i0, j0);

    // clamp in x only: 1D interp along y at boundary i0
    if (i0 == i1 && j0 != j1) {
      return maths::linear_interpolation(y, grid1[j0], metad_potential_(i0, j0),
                                            grid1[j1], metad_potential_(i0, j1));
    }
    // clamp in y only: 1D interp along x at boundary j0
    if (j0 == j1 && i0 != i1) {
      return maths::linear_interpolation(x, grid0[i0], metad_potential_(i0, j0),
                                            grid0[i1], metad_potential_(i1, j0));
    }
    // interior: bilinear
    const double Q00 = metad_potential_(i0, j0);
    const double Q01 = metad_potential_(i0, j1);
    const double Q10 = metad_potential_(i1, j0);
    const double Q11 = metad_potential_(i1, j1);
    return maths::bilinear_interpolation(x, y, grid0[i0], grid1[j0], grid0[i1], grid1[j1], Q00, Q01, Q10, Q11);
  }

  assert(false && "Unreachable code: base_potential only supports up to kNumCVars dimensions");
  UNREACHABLE();
}

// Returns the nearest grid indices to the given cvar_coordinates. If the coordinates are outside of the grid then
// the index clamps to the nearest edge.
std::array<int, jams::MetadynamicsPotential::kNumCVars>
jams::MetadynamicsPotential::potential_grid_indices(const std::array<double, kNumCVars> &cvar_coordinates) {
  std::array<int, kNumCVars> idx{};
  for (auto n = 0; n < cvars_.size(); ++n) {
    const double x   = cvar_coordinates[n];
    const double min = cvar_range_min_[n];
    const double max = cvar_range_max_[n];
    const double inv_step = cvar_inv_step_[n]; // = 1.0 / range_step

    const int npoints = num_cvar_sample_coordinates_[n];

    // Map into grid coordinate (0 .. npoints-1) in floating point
    double t = (x - min) * inv_step;

    // Floor to integer cell index
    int i = static_cast<int>(std::floor(t));

    // Clamp to [0, npoints-1]
    if (i < 0) {
      i = 0;
    } else if (i >= npoints) {
      i = npoints - 1;
    }

    idx[n] = i;
  }
  return idx;
}

void jams::MetadynamicsPotential::insert_gaussian(double relative_amplitude) {

  auto center = cvar_coordinates();

  for (auto n = 0; n < center.size(); ++n) {
    if (!std::isfinite(center[n])) {
      throw std::runtime_error("Collective variable value is not finite for CV '" + cvar_names_[n] + "'");
    }
  }
  // Suppress deposition outside the configured range depending on the active BCs.
  // - HardBC: suppress outside range
  // - RestoringBC: suppress only when outside range (allowed within range even if beyond threshold)
  // - MirrorBC: do not suppress; mirrored Gaussians will be inserted below
  // - NoBC: suppress outside range (undefined behaviour otherwise)
  bool suppress = false;
  for (auto n = 0; n < cvars_.size(); ++n) {
    const double lo = cvar_range_min_[n];
    const double hi = cvar_range_max_[n];
    if (center[n] < lo) {
      auto bc = cvar_lower_bcs_[n];
      if (bc == PotentialBCs::MirrorBC) {
        continue; // allow mirroring
      } else {
        suppress = true; break;
      }
    } else if (center[n] > hi) {
      auto bc = cvar_upper_bcs_[n];
      if (bc == PotentialBCs::MirrorBC) {
        continue; // allow mirroring
      } else {
        suppress = true; break;
      }
    }
  }
  if (suppress) {
    relative_amplitude = 0.0;
  }

  MultiArray<double,kNumCVars> metad_potential_update(metad_potential_.shape());
  metad_potential_update.zero();

  add_gaussian_to_landscape(center, metad_potential_update);

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
      if (std::abs(center[n] - mirror_bc_lower_threshold_[n]) <= kGaussianExtent * cvar_gaussian_widths_[n]) {
        auto virtual_center = center;
        virtual_center[n] = 2*mirror_bc_lower_threshold_[n] - virtual_center[n];
        add_gaussian_to_landscape(virtual_center, metad_potential_update);
      }
    }

    if (cvar_upper_bcs_[n] == PotentialBCs::MirrorBC) {
      if (std::abs(center[n] - mirror_bc_upper_threshold_[n]) <= kGaussianExtent * cvar_gaussian_widths_[n]) {
        auto virtual_center = center;
        virtual_center[n] = 2*mirror_bc_upper_threshold_[n] - virtual_center[n];
        add_gaussian_to_landscape(virtual_center, metad_potential_update);
      }
    }
  }

  // Calculate the total gaussian mass deposited
  double gaussian_mass = 0.0;
  for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
    for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
      gaussian_mass += metad_potential_update(i, j);
    }
  }

  if (!std::isfinite(gaussian_mass)) {
    throw std::runtime_error("Gaussian mass is not finite.");
  }

  // Adjust the amplitude by normalising the total gaussian mass deposited
  // to 1 and multiplying by the basic gaussian amplitude and the relative
  // amplitude from any tempering.
  if (gaussian_mass != 0.0) {
    const double amplitude_adjustment = relative_amplitude * metad_gaussian_amplitude_ / gaussian_mass;
    for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
      for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
        metad_potential_update(i, j) *= amplitude_adjustment;
      }
    }
  }

  // Update the metadynamics potential
  for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
    for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
      metad_potential_(i, j) += metad_potential_update(i, j);
    }
  }

  // Update the potential that tracks changes between outputs to the synchronisation file
  for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
    for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
      metad_potential_delta_(i, j) += metad_potential_update(i, j);
    }
  }

  if (globals::solver->iteration() % cvar_output_stride_ == 0 ) {
    cvar_output_file_ << globals::solver->time();
    for (auto n = 0; n < cvars_.size(); ++n) {
      cvar_output_file_ << " " << cvars_[n]->value();
    }
    cvar_output_file_ << " " << relative_amplitude << std::endl;
  }
}


std::array<double, jams::MetadynamicsPotential::kNumCVars>
jams::MetadynamicsPotential::cvar_coordinates() {
  std::array<double,kNumCVars> coordinates{};
  for (auto n = 0; n < cvars_.size(); ++n) {
    coordinates[n] = cvars_[n]->value();
  }
  return coordinates;
}

void jams::MetadynamicsPotential::output(const std::string &filename) const {
  std::ofstream of(filename);

  for (auto n = 0; n < cvars_.size(); ++n) {
    of << cvar_names_[n] << " ";
  }

  of << "potential_meV" << "\n";

  assert(cvars_.size() <= kNumCVars);
  if (cvars_.size() == 1) {
    for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
      of << cvar_sample_coordinates_[0][i] << " " << metad_potential_(i, 0) << "\n";
    }
    return;
  }

  if (cvars_.size() == 2) {
    for (auto i = 0; i < num_cvar_sample_coordinates_[0]; ++i) {
      for (auto j = 0; j < num_cvar_sample_coordinates_[1]; ++j) {
        of << cvar_sample_coordinates_[0][i] << " " << cvar_sample_coordinates_[1][j] << " " << metad_potential_(i, j) << "\n";
      }
    }
    return;
  }
  assert(false); // Should not be reachable if cvars_.size() <= kNumCVars
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

          if (!std::isfinite(potential_passed)) {
            throw std::runtime_error("Non-finite potential value on line " + std::to_string(line_number));
          }

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

          if (!std::isfinite(potential_passed)) {
            throw std::runtime_error("Non-finite potential value on line " + std::to_string(line_number));
          }

          if (is.bad() || is.fail()) {
            throw std::runtime_error("failed to read line " + std::to_string(line_number));
          }

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
              metad_potential_(i, j) = file_data[copy_iterator++];
            }
        }
        potential_file_passed.close();
    }
}

void jams::MetadynamicsPotential::synchronise_shared_potential(const std::string &file_name) {
  auto lock_file_name = file_name + ".lock";
  int lock_fd = output::lock_file(lock_file_name);

  MultiArray<double, kNumCVars> shared_potential(metad_potential_.shape());
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
void jams::MetadynamicsPotential::print_settings() const {
  std::cout << "metad_gaussian_amplitude: " << metad_gaussian_amplitude_ << "\n";
  std::cout << "cvar_output_stride: " << cvar_output_stride_ << "\n";
  for (auto n = 0; n < cvars_.size(); ++n) {
    std::cout << "cvar_names[" << n << "]: " << cvar_names_[n] << "\n";
    std::cout << "cvar_range_min[" << n << "]: " << cvar_range_min_[n] << "\n";
    std::cout << "cvar_range_max[" << n << "]: " << cvar_range_max_[n] << "\n";
    std::cout << "cvar_lower_bcs[" << n << "]: " << cvar_lower_bcs_[n] << "\n";
    std::cout << "cvar_upper_bcs[" << n << "]: " << cvar_upper_bcs_[n] << "\n";
    std::cout << "restoring_bc_lower_threshold[" << n << "]: " << restoring_bc_lower_threshold_[n] << "\n";
    std::cout << "restoring_bc_upper_threshold[" << n << "]: " << restoring_bc_upper_threshold_[n] << "\n";
    std::cout << "restoring_bc_spring_constant[" << n << "]: " << restoring_bc_spring_constant_[n] << "\n";
    std::cout << "cvar_gaussian_widths[" << n << "]: " << cvar_gaussian_widths_[n] << "\n";
    std::cout << "num_cvar_sample_coordinates[" << n << "]: " << num_cvar_sample_coordinates_[n] << "\n";
  }
}



