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
tsv_file_(jams::output::full_path_filename("jsp.tsv"))
{
  tsv_file_.setf(std::ios::right);
  tsv_file_ << tsv_header();

  material_count_.resize(globals::lattice->num_materials(), 0);
  for (auto i = 0; i < globals::num_spins; ++i) {
    material_count_[globals::lattice->atom_material_id(i)]++;
  }
}

void SpinPumpingMonitor::update(Solver * solver) {
  tsv_file_.width(12);

  std::vector<Vec3> spin_pumping_real(material_count_.size());
  std::vector<Vec3> spin_pumping_imag(material_count_.size());

  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto type = globals::lattice->atom_material_id(i);

    Vec3 s_i = {globals::s(i,0), globals::s(i, 1), globals::s(i,2)};
    Vec3 ds_dt_i = {globals::ds_dt(i,0), globals::ds_dt(i, 1), globals::ds_dt(i,2)};

    spin_pumping_real[type] += cross(s_i, ds_dt_i);
    spin_pumping_imag[type] += ds_dt_i;
  }

  // output in rad / s^-1 T^-1
  tsv_file_ << std::scientific << solver->time() << "\t";

  for (auto type = 0; type < material_count_.size(); ++type) {
    auto norm = 1.0 / static_cast<double>(material_count_[type]);
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
  for (auto i = 0; i < globals::lattice->num_materials(); ++i) {
    auto name = globals::lattice->material_name(i);
    ss << name + "_Re_J_x\t";
    ss << name + "_Re_J_y\t";
    ss << name + "_Re_J_z\t";
    ss << name + "_Im_J_x\t";
    ss << name + "_Im_J_y\t";
    ss << name + "_Im_J_z\t";
  }
  ss << std::endl;

  return ss.str();
}
