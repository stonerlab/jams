// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_GLOBALS_H
#define JAMS_CORE_GLOBALS_H

#include <string>
#include <memory>

#include "jams/containers/multiarray.h"
#include "jams/helpers/mixed_precision.h"

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif

class Solver;
class Lattice;
namespace libconfig { class Config; }

///< Config object

namespace globals {
  GLOBAL int num_spins;
  GLOBAL int num_spins3;

  /// @brief global spin data where spins are unit vectors
  ///
  /// @details s(site_index, cartesian_component)
  /// - site_index: enumerates sites in the whole lattice
  /// - cartesian_component: 0: x | 1: y | 2: z
  GLOBAL jams::MultiArray<double, 2> s;

  /// @brief global local field data (H_eff) in units of Tesla / mu_s
  ///
  /// @details h(site_index, cartesian_component)
  /// - site_index: enumerates sites in the whole lattice
  /// - cartesian_component: 0: x | 1: y | 2: z
  GLOBAL jams::MultiArray<jams::Real, 2> h;

  /// @brief global angular velocity data ds_dt
  ///
  /// @details ds_dt(site_index, cartesian_component)
  /// - site_index: enumerates sites in the whole lattice
  /// - cartesian_component: 0: x | 1: y | 2: z
  GLOBAL jams::MultiArray<double, 2> ds_dt;

  /// @brief global lattice cartesian position data in units of lattice constants
  ///
  /// @details positions(site_index, cartesian_component)
  /// - site_index: enumerates sites in the whole lattice
  /// - cartesian_component: 0: x | 1: y | 2: z
  GLOBAL jams::MultiArray<jams::Real, 2> positions;

  /// @brief global dimensionless damping constant data
  ///
  /// @details positions(site_index)
  /// - site_index: enumerates sites in the whole lattice
  GLOBAL jams::MultiArray<jams::Real, 1> alpha;

  /// @brief global magnetic moment data in units of Joules/Tesla
  ///
  /// @details positions(site_index)
  /// - site_index: enumerates sites in the whole lattice
  GLOBAL jams::MultiArray<jams::Real, 1> mus;

  /// @brief global reciprocal magnetic moment data in Tesla/Joules
  ///
  /// @details Zero-moment vacancy sites have reciprocal moment zero.
  GLOBAL jams::MultiArray<jams::Real, 1> inv_mus;

  /// @brief number of sites with a non-zero magnetic moment
  GLOBAL int num_magnetic_spins;

  /// @brief global gyromagnetic moment data in units of gyromagnetic ratios
  ///
  /// @details positions(site_index)
  /// - site_index: enumerates sites in the whole lattice
  GLOBAL jams::MultiArray<jams::Real, 1> gyro;
  
  GLOBAL Solver  *solver;
  GLOBAL Lattice *lattice;
  GLOBAL std::unique_ptr<libconfig::Config> config;
  GLOBAL std::string simulation_name;

  void sync_magnetic_moment_data();
  int first_magnetic_spin();
}  // namespace globals
#undef GLOBAL
#endif  // JAMS_CORE_GLOBALS_H
