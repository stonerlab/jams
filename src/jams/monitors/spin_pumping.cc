// Copyright 2014 Joseph Barker. All rights reserved.


#include "jams/monitors/spin_pumping.h"

#include <string>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>

#include "jams/core/consts.h"
#include "jams/core/lattice.h"
#include "jams/core/output.h"
#include "jams/core/solver.h"
#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/core/stats.h"
#include "jblib/containers/array.h"

SpinPumpingMonitor::SpinPumpingMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  ::output->write("Initialising Energy Distribution monitor\n");

  convergence_is_on_ = false;
  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
    ::output->write("  convergence tolerance: %f\n", convergence_tolerance_);
  }

  // std::string name = "_iz_dist.tsv";
  // name = seedname+name;
  // iz_dist_file.open(name.c_str());
  // iz_dist_file.setf(std::ios::right);
  // iz_dist_file << std::setw(12) << "bin" << "\t";
  // iz_dist_file << std::setw(12) << "Iz_g_real" << "\t";
  // iz_dist_file << std::setw(12) << "count" << std::endl;


  std::string filename = "_iz_mean.tsv";
  filename = seedname+filename;
  iz_mean_file.open(filename.c_str());
  iz_mean_file.setf(std::ios::right);
  iz_mean_file << std::setw(12) << "time" << "\t";
  iz_mean_file << std::setw(12) << "Iz_g_real" << "\t";
  iz_mean_file << std::setw(12) << "Iz_g_imag" << "\t";

  if (convergence_is_on_) {
    iz_mean_file << std::setw(12) << "geweke";
  }

  iz_mean_file << std::endl;

  // name = "_w_dist.tsv";
  // name = seedname+name;
  // w_dist_file.open(name.c_str());
  // w_dist_file << std::setw(12) << "bin" << "\t";
  // w_dist_file << std::setw(12) << "w_freq" << "\t";
  // w_dist_file << std::setw(12) << "count" << std::endl;

}

void SpinPumpingMonitor::update(Solver * solver) {
  using namespace globals;
  using std::abs;

    std::vector<Stats> spin_pumping_re(::lattice->num_materials());
    std::vector<Stats> spin_pumping_im(::lattice->num_materials());

    for (int i = 0; i < num_spins; ++i) {
      spin_pumping_re[::lattice->atom_material(i)].add((s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0)));
      spin_pumping_im[::lattice->atom_material(i)].add(ds_dt(i, 2));
    }

    // output in rad / s^-1 T^-1
    iz_mean_file << std::setw(12) << std::scientific << solver->time() << "\t";

    for (int n = 0; n < ::lattice->num_materials(); ++n) {
      iz_mean_file << std::setw(12) << std::scientific << spin_pumping_re[n].mean() * kGyromagneticRatio << "\t";
      iz_mean_file << std::setw(12) << std::scientific << spin_pumping_im[n].mean() * kGyromagneticRatio << "\t";
    }


    // if (convergence_is_on_) {
    //   convergence_stats_.add(pumping.mean());
    //   convergence_geweke_diagnostic_ = convergence_stats_.geweke();
    //   iz_mean_file << std::setw(12) << convergence_geweke_diagnostic_;
    // }

    iz_mean_file << std::endl;

    // Stats angular_velocity_stats(angular_velocity_total);
    // angular_velocity_stats.histogram(range, bin);

    // norm = 1.0 / angular_velocity_total.size();
    // for (int i = 0; i < bin.size(); ++i) {
    //   // output in Hz
    //   w_dist_file << std::setw(12) << i << "\t";
    //   w_dist_file << std::setw(12) << std::scientific << 0.5*(range[i] + range[i+1]) * kGyromagneticRatio << "\t";
    //   w_dist_file << std::setw(12) << std::scientific << bin[i] * norm << "\n";
    // }
    // w_dist_file << "\n" << std::endl;

}

bool SpinPumpingMonitor::is_converged() {
  return ((std::abs(convergence_geweke_diagnostic_) < convergence_tolerance_) && convergence_is_on_);
}

SpinPumpingMonitor::~SpinPumpingMonitor() {
  // w_dist_file.close();
  // iz_dist_file.close();
  iz_mean_file.close();
}
