// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/interface/openmp.h"
#include "jams/helpers/output.h"

#include "jams/monitors/magnetisation.h"
#include "jams/helpers/spinops.h"
#include "jams/helpers/array_ops.h"


MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("mag.tsv"))
{
  // calculate magnetisation per material or per unit cell position
  auto grouping_str = lowercase(
    jams::config_optional<std::string>(settings, "grouping", "materials"));

  if (grouping_str == "none") {
    grouping_ = Grouping::NONE;
  } else if (grouping_str == "materials") {
    grouping_ = Grouping::MATERIALS;
  } else if (grouping_str == "positions") {
    grouping_ = Grouping::POSITIONS;
  } else {
    throw std::runtime_error("unknown magnetisation grouping: " + grouping_str);
  }

  // should the magnetisation be normalised to 1 or be in units of muB
  normalize_magnetisation_ = jams::config_optional<bool>(settings, "normalize", true);


  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();

  switch (grouping_) {
    case Grouping::NONE: {
      jams::MultiArray<int,1> indices(globals::num_spins);
      for (auto i = 0; i < globals::num_spins; ++i) {
        indices(i) = i;
      }
      group_spin_indicies_.push_back(indices);
      break;
    }

    case Grouping::MATERIALS: {
      std::vector<std::vector<int>> material_index_groups(globals::lattice->num_materials());
      for (auto i = 0; i < globals::num_spins; ++i) {
        auto type = globals::lattice->lattice_site_material_id(i);
        material_index_groups[type].push_back(i);
      }

      group_spin_indicies_.resize(material_index_groups.size());
      for (auto n = 0; n < material_index_groups.size(); ++n) {
        group_spin_indicies_[n] = jams::MultiArray<int,1>(
            material_index_groups[n].begin(),
            material_index_groups[n].end());
      }
      break;
    }

    case Grouping::POSITIONS: {
      std::vector<std::vector<int>> position_index_groups(globals::lattice->num_basis_sites());
      for (auto i = 0; i < globals::num_spins; ++i) {
        auto pos = globals::lattice->lattice_site_basis_index(i);
        position_index_groups[pos].push_back(i);
      }

      group_spin_indicies_.resize(position_index_groups.size());
      for (auto n = 0; n < position_index_groups.size(); ++n) {
        group_spin_indicies_[n] = jams::MultiArray<int,1>(
            position_index_groups[n].begin(),
            position_index_groups[n].end());
      }
      break;
    }

    default:
      break;
  }
}

void MagnetisationMonitor::update(Solver& solver) {
  using namespace jams;

  tsv_file.width(12);
  tsv_file << fmt::sci << solver.time();

  for (auto n = 0; n < group_spin_indicies_.size(); ++n) {
    Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus, group_spin_indicies_[n]);
    double normalising_factor = 1.0;
    if (normalize_magnetisation_) {
      normalising_factor = 1.0 / jams::scalar_field_indexed_reduce(globals::mus, group_spin_indicies_[n]);
    } else {
      // internally we use meV T^-1 for mus so convert back to Bohr magneton
      normalising_factor = 1.0 / kBohrMagnetonIU;
    }
    tsv_file << fmt::sci << mag[0] * normalising_factor;
    tsv_file << fmt::sci << mag[1] * normalising_factor;
    tsv_file << fmt::sci << mag[2] * normalising_factor;
    tsv_file << fmt::sci << norm(mag) * normalising_factor;
  }

  tsv_file << std::endl;
}

std::string MagnetisationMonitor::tsv_header() {
  using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss << fmt::sci << "time";

  switch (grouping_) {
    case Grouping::NONE: {
      for (const auto &name : {"mx", "my", "mz", "m"}) {
        ss << fmt::sci << name;
      }
      break;
    }

    case Grouping::MATERIALS: {
      const auto nm = globals::lattice->num_materials();
      for (int i = 0; i < nm; ++i) {
        auto name = globals::lattice->material_name(i);
        for (const auto &suffix : {"_mx", "_my", "_mz", "_m"}) {
          ss << fmt::sci << name + suffix;
        }
      }
      break;
    }

    case Grouping::POSITIONS: {
      const auto nb = globals::lattice->num_basis_sites();
      for (int i = 0; i < nb; ++i) {
        const auto mat = globals::lattice->basis_site_atom(i).material_index;
        auto material_name = globals::lattice->material_name(mat);
        for (const auto &suffix : {"_mx", "_my", "_mz", "_m"}) {
          ss << fmt::sci << std::to_string(i + 1) + "_" + material_name + suffix;
        }
      }
      break;
    }

    default:
      break;
  }

  ss << std::endl;

  return ss.str();
}
