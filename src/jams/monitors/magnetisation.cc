// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/interface/openmp.h"
#include "jams/helpers/output.h"

#include "jams/monitors/magnetisation.h"
#include "magnetisation.h"
#include "jams/helpers/spinops.h"
#include "jams/helpers/array_ops.h"


MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("mag.tsv"))
{
  // calculate magnetisation per material or per unit cell position
  auto grouping_str = jams::config_optional<std::string>(settings, "grouping", "materials");

  if (lowercase(grouping_str) == "materials") {
    grouping_ = Grouping::MATERIALS;
  } else if (lowercase(grouping_str) == "positions") {
    grouping_ = Grouping::POSITIONS;
  } else {
    throw std::runtime_error("unknown magnetisation grouping: " + grouping_str);
  }

  // should the magnetisation be normalised to 1 or be in units of muB
  normalize_magnetisation_ = jams::config_optional<bool>(settings, "normalize", true);


  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();


  if (grouping_ == Grouping::MATERIALS) {
    std::vector<std::vector<int>> material_index_groups(globals::lattice->num_materials());
    for (auto i = 0; i < globals::num_spins; ++i) {
      auto type = globals::lattice->atom_material_id(i);
      material_index_groups[type].push_back(i);
    }

    group_spin_indicies_.resize(material_index_groups.size());
    for (auto n = 0; n < material_index_groups.size(); ++n) {
      group_spin_indicies_[n] = jams::MultiArray<int,1>(material_index_groups[n].begin(), material_index_groups[n].end());
    }
  } else if (grouping_ == Grouping::POSITIONS) {
    std::vector<std::vector<int>> position_index_groups(globals::lattice->num_motif_atoms());
    for (auto i = 0; i < globals::num_spins; ++i) {
      auto position = globals::lattice->atom_motif_position(i);
      position_index_groups[position].push_back(i);
    }

    group_spin_indicies_.resize(position_index_groups.size());
    for (auto n = 0; n < position_index_groups.size(); ++n) {
      group_spin_indicies_[n] = jams::MultiArray<int,1>(position_index_groups[n].begin(), position_index_groups[n].end());
    }
  }
}

void MagnetisationMonitor::update(Solver& solver) {
  using namespace jams;

  tsv_file.width(12);
  tsv_file << fmt::sci << solver.time();
  tsv_file << fmt::decimal << solver.physics()->temperature();

  for (auto i = 0; i < 3; ++i) {
    tsv_file << fmt::decimal << solver.physics()->applied_field(i);
  }

  for (auto n = 0; n < group_spin_indicies_.size(); ++n) {
    Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus, group_spin_indicies_[n]);
    double normalising_factor = 1.0;
    if (normalize_magnetisation_) {
      normalising_factor = 1.0 / jams::scalar_field_indexed_reduce(globals::mus, group_spin_indicies_[n]);
    } else {
      // internally we use meV T^-1 for mus so convert back to Bohr magneton
      normalising_factor = 1.0 / kBohrMagnetonIU;
    }
    tsv_file << fmt::decimal << mag[0] * normalising_factor;
    tsv_file << fmt::decimal << mag[1] * normalising_factor;
    tsv_file << fmt::decimal << mag[2] * normalising_factor;
    tsv_file << fmt::decimal << norm(mag) * normalising_factor;
  }

  tsv_file << std::endl;
}

std::string MagnetisationMonitor::tsv_header() {
  using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss << fmt::sci << "time";
  ss << fmt::decimal << "T";
  ss << fmt::decimal << "hx";
  ss << fmt::decimal << "hy";
  ss << fmt::decimal << "hz";

  if (grouping_ == Grouping::MATERIALS) {
    for (auto i = 0; i < globals::lattice->num_materials(); ++i) {
      auto name = globals::lattice->material_name(i);
      for (const auto &suffix: {"_mx", "_my", "_mz", "_m"}) {
        ss << fmt::decimal << name + suffix;
      }
    }
  } else if (grouping_ == Grouping::POSITIONS) {
    for (auto i = 0; i < globals::lattice->num_motif_atoms(); ++i) {
      auto material_name = globals::lattice->material_name(
          globals::lattice->motif_atom(i).material_index);
      for (const auto &suffix: {"_mx", "_my", "_mz", "_m"}) {
        ss << fmt::decimal << std::to_string(i+1) + "_" + material_name + suffix;
      }
    }
  }

  ss << std::endl;

  return ss.str();
}
