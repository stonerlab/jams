// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>
#include <iostream>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

#include "smr.h"

SMRMonitor::SMRMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("smr.tsv")),
  spin_groups_(jams::monitors::make_spin_groups(jams::monitors::SpinGrouping::MATERIALS))
{
  std::cout << "\ninitialising SMR monitor\n";
  std::cout << "  assumes axes j->x, t->y, n->z\n";

  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void SMRMonitor::update(Solver& solver) {
  const auto spins = globals::s.host_view();

  std::vector<double> mtsq_para(spin_groups_.size(), 0.0);
  std::vector<double> mtsq_perp(spin_groups_.size(), 0.0);

  std::vector<double> mjmt_para(spin_groups_.size(), 0.0);
  std::vector<double> mjmt_perp(spin_groups_.size(), 0.0);

  std::vector<double> mn(spin_groups_.size(), 0.0);

  for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
    // Uses the WMI geometry from M. Althammer,Phys. Rev. B 87, 224401 (2013).
    // assuming axes:
    // j -> x
    // t -> y
    // n -> z
    for (const auto i : spin_groups_[group_index].indices) {
      mtsq_para[group_index] +=  spins(i, 1) * spins(i, 1);
      mtsq_perp[group_index] +=  spins(i, 0) * spins(i, 0);
      mjmt_para[group_index] +=  spins(i, 0) * spins(i, 1);
      mjmt_perp[group_index] += -spins(i, 0) * spins(i, 1);

      mn[group_index] += spins(i, 2);
    }
  }

  for (std::size_t i = 0; i < spin_groups_.size(); ++i) {
    if (!spin_groups_[i].indices.empty()) {
      const auto norm = 1.0 / static_cast<double>(spin_groups_[i].indices.size());
      mtsq_para[i] = mtsq_para[i] * norm;
      mtsq_perp[i] = mtsq_perp[i] * norm;

      mjmt_para[i] = mjmt_para[i] * norm;
      mjmt_perp[i] = mjmt_perp[i] * norm;

      mn[i] = mn[i] * norm;
    }
  }

  tsv_file << std::setw(12) << std::scientific << solver.time() << "\t";

  for (std::size_t i = 0; i < spin_groups_.size(); ++i) {
    tsv_file << std::setw(12) << mtsq_para[i] << "\t" << mtsq_perp[i] << "\t";
    tsv_file << std::setw(12) << mjmt_para[i] << "\t" << mjmt_perp[i] << "\t";
    tsv_file << std::setw(12) << mn[i] << "\t";
  }

  tsv_file << std::endl;
}

std::string SMRMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (const auto& group : spin_groups_) {
    ss << group.name + "_mtsq_para" << "\t";
    ss << group.name + "_mtsq_perp" << "\t";
    ss << group.name + "_mjmt_para" << "\t";
    ss << group.name + "_mjmt_perp" << "\t";
    ss << group.name + "_mn" << "\t";
  }

  ss << std::endl;

  return ss.str();
}
