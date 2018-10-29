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
#include "jblib/containers/array.h"

SpinPumpingMonitor::SpinPumpingMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  tsv_file.open(seedname + "_iz_mean.tsv");
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void SpinPumpingMonitor::update(Solver * solver) {
  using namespace globals;
  using std::abs;

  tsv_file.width(12);

  std::vector<Stats> spin_pumping_re(::lattice->num_materials());
  std::vector<Stats> spin_pumping_im(::lattice->num_materials());

  for (int i = 0; i < num_spins; ++i) {
    spin_pumping_re[::lattice->atom_material_id(i)].add((s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0)));
    spin_pumping_im[::lattice->atom_material_id(i)].add(ds_dt(i, 2));
  }

  // output in rad / s^-1 T^-1
  tsv_file << std::scientific << solver->time() << "\t";

  for (int n = 0; n < ::lattice->num_materials(); ++n) {
    tsv_file << std::scientific << spin_pumping_re[n].mean() * kGyromagneticRatio << "\t";
    tsv_file << std::scientific << spin_pumping_im[n].mean() * kGyromagneticRatio << "\t";
  }

  tsv_file << std::endl;
}

std::string SpinPumpingMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  ss << "Iz_g_real\t";
  ss << "Iz_g_imag\t";

  ss << std::endl;

  return ss.str();
}
