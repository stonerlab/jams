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
#include "jams/helpers/container_utils.h"


MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_(make_tsv_writer(settings))
{}

void MagnetisationMonitor::update(Solver& solver) {
  auto values = make_reserved<double>(tsv_.num_cols());

  values.push_back(solver.time());

  for (auto n = 0; n < group_spin_indices_.size(); ++n) {
    Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus, group_spin_indices_[n]);
    double normalising_factor = 1.0;
    if (normalize_magnetisation_) {
      normalising_factor = 1.0 / jams::scalar_field_indexed_reduce(globals::mus, group_spin_indices_[n]);
    } else {
      // internally we use meV T^-1 for mus so convert back to Bohr magneton
      normalising_factor = 1.0 / kBohrMagnetonIU;
    }

    values.push_back(mag[0] * normalising_factor);
    values.push_back(mag[1] * normalising_factor);
    values.push_back(mag[2] * normalising_factor);
    values.push_back(jams::norm(mag) * normalising_factor);
  }

  tsv_.write_row(values);
}


jams::output::TsvWriter MagnetisationMonitor::make_tsv_writer(const libconfig::Setting &settings) {
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

  switch (grouping_) {
    case Grouping::NONE: {
      jams::MultiArray<int,1> indices(globals::num_spins);
      for (auto i = 0; i < globals::num_spins; ++i) {
        indices(i) = i;
      }
      group_spin_indices_.push_back(indices);
      break;
    }

    case Grouping::MATERIALS: {
      std::vector<std::vector<int>> material_index_groups(globals::lattice->num_materials());
      for (auto i = 0; i < globals::num_spins; ++i) {
        auto type = globals::lattice->lattice_site_material_id(i);
        material_index_groups[type].push_back(i);
      }

      group_spin_indices_.resize(material_index_groups.size());
      for (auto n = 0; n < material_index_groups.size(); ++n) {
        group_spin_indices_[n] = jams::MultiArray<int,1>(
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

      group_spin_indices_.resize(position_index_groups.size());
      for (auto n = 0; n < position_index_groups.size(); ++n) {
        group_spin_indices_[n] = jams::MultiArray<int,1>(
            position_index_groups[n].begin(),
            position_index_groups[n].end());
      }
      break;
    }

    default:
      break;
  }

  auto precision = jams::config_optional<int>(settings, "precision", 8);
  std::vector<jams::output::ColDef> cols;

  std::string mag_unit = "dimensionless";
  if (!normalize_magnetisation_) {
    mag_unit = "bohr magnetons";
  }

  cols.push_back({"time", "picoseconds"});

  switch (grouping_) {
    case Grouping::NONE: {
      for (const auto &name : {"mx", "my", "mz", "m"}) {
        cols.push_back({name, mag_unit});
      }
      break;
    }

    case Grouping::MATERIALS: {
      const auto nm = globals::lattice->num_materials();
      for (int i = 0; i < nm; ++i) {
        auto name = globals::lattice->material_name(i);
        for (const auto &suffix : {"_mx", "_my", "_mz", "_m"}) {
          cols.push_back({name + suffix, mag_unit});
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
          cols.push_back({material_name + "_" + std::to_string(i + 1) + suffix, mag_unit});
        }
      }
      break;
    }

    default:
      break;
  }

  return jams::output::TsvWriter(
    jams::output::full_path_filename("mag.tsv"),
    std::move(cols),
    precision
  );
}
