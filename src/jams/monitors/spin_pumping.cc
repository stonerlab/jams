// Copyright 2014 Joseph Barker. All rights reserved.


#include "spin_pumping.h"

#include <string>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/helpers/stats.h"
#include "jams/helpers/output.h"

SpinPumpingMonitor::SpinPumpingMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file_(jams::output::full_path_filename("jsp.tsv")),
  spin_groups_(jams::monitors::make_spin_groups(jams::monitors::SpinGrouping::MATERIALS))
{
  tsv_file_.setf(std::ios::right);
  tsv_file_ << tsv_header();

  s_old_.resize(globals::num_spins,3);
}

void SpinPumpingMonitor::update(Solver& solver) {
  tsv_file_.width(12);
  const auto spins = globals::s.host_view();

  std::vector<jams::Vec<double, 3>> spin_pumping_real(spin_groups_.size());
  std::vector<jams::Vec<double, 3>> spin_pumping_imag(spin_groups_.size());
  double d_timestep = 1.0/solver.time_step();

  for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
    for (const auto i : spin_groups_[group_index].indices) {
      jams::Vec<double, 3> s_i = {spins(i,0), spins(i, 1), spins(i,2)};
      jams::Vec<double, 3> s_old_i = {s_old_(i,0), s_old_(i, 1), s_old_(i,2)};
      jams::Vec<double, 3> ds_dt_i = (s_i - s_old_i) * d_timestep;

      spin_pumping_real[group_index] += jams::cross(s_i, ds_dt_i);
      spin_pumping_imag[group_index] += ds_dt_i;
    }
  }

  // output in rad / s^-1 T^-1
  tsv_file_ << std::scientific << solver.time() << "\t";

  for (std::size_t type = 0; type < spin_groups_.size(); ++type) {
    auto norm = spin_groups_[type].indices.empty()
        ? 0.0
        : 1.0 / static_cast<double>(spin_groups_[type].indices.size());
    for (auto j = 0; j < 3; ++j) {
      tsv_file_ << std::scientific << spin_pumping_real[type][j] * norm  << "\t";
    }
    for (auto j = 0; j < 3; ++j) {
      tsv_file_ << std::scientific << spin_pumping_imag[type][j] * norm << "\t";
    }
  }

  tsv_file_ << std::endl;
}

std::string SpinPumpingMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (const auto& group : spin_groups_) {
    ss << group.name + "_Re_J_x\t";
    ss << group.name + "_Re_J_y\t";
    ss << group.name + "_Re_J_z\t";
    ss << group.name + "_Im_J_x\t";
    ss << group.name + "_Im_J_y\t";
    ss << group.name + "_Im_J_z\t";
  }
  ss << std::endl;

  return ss.str();
}


bool SpinPumpingMonitor::is_updating(const int &iteration) {
  if ((iteration + 1) % output_step_freq_ == 0) {
    const auto& spins = globals::s;
    s_old_ = spins;
  }
  if (iteration % output_step_freq_ == 0) {
    return true;
  }
  return false;
}
