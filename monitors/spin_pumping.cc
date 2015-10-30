// Copyright 2014 Joseph Barker. All rights reserved.

#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

#include "monitors/spin_pumping.h"

#include <string>

#include "core/globals.h"
#include "core/maths.h"
#include "core/stats.h"
#include "core/hamiltonian.h"

SpinPumpingMonitor::SpinPumpingMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  ::output.write("Initialising Energy Distribution monitor\n");

  std::string name = "_iz_dist.dat";
  name = seedname+name;
  iz_dist_file.open(name.c_str());

  name = "_iz_mean.dat";
  name = seedname+name;
  iz_mean_file.open(name.c_str());

  name = "_w_dist.dat";
  name = seedname+name;
  w_dist_file.open(name.c_str());

}

void SpinPumpingMonitor::update(Solver * solver) {
  using namespace globals;



  if (solver->iteration()%output_step_freq_ == 0) {
    // for (std::vector<Hamiltonian*>::iterator it = solver->hamiltonians().begin() ; it != solver->hamiltonians().end(); ++it) {
    //   (*it)->calculate_energies();
    // }

    // // sum hamiltonian energy contributions
    // for (std::vector<Hamiltonian*>::iterator it = solver->hamiltonians().begin() ; it != solver->hamiltonians().end(); ++it) {
    //   cblas_daxpy(num_spins, 1.0, (*it)->ptr_energy(), 1, &energy_total[0], 1);
    // }

    // Stats energy_stats(energy_total);

    // std::vector<double> range;
    // std::vector<double> bin;

    // energy_stats.histogram(range, bin);

    // for (int i = 0; i < bin.size(); ++i) {
    //   outfile << i << "\t" << 0.5*(range[i] + range[i+1]) * kBohrMagneton << "\t" << bin[i] << "\n";
    // }
    // outfile << "\n" << std::endl;

    std::vector<double> energy_total(num_spins, 0.0);
    std::vector<double> pumping_total(num_spins, 0.0);
    std::vector<double> angular_velocity_total(num_spins, 0.0);


    for (int i = 0; i < num_spins; ++i) {
      // double vt[3]= { 0.0,
      //                 0.0,
      //                 s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0) };

      pumping_total[i] = ((s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0)));
      angular_velocity_total[i] = ((s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0))) / (s(i, 0)*s(i, 0) + s(i, 1)*s(i, 1));
    }

    std::vector<double> range;
    std::vector<double> bin;

    Stats pumping(pumping_total);

    pumping.histogram(range, bin);

    double norm = 1.0 / pumping_total.size();
    for (int i = 0; i < bin.size(); ++i) {
      iz_dist_file << i << "\t" << 0.5*(range[i] + range[i+1]) << "\t" << bin[i] * norm << "\n";
    }
    iz_dist_file << "\n" << std::endl;

    iz_mean_file << solver->time() << "\t" << pumping.mean() << std::endl;

    std::vector<double> w_range;
    std::vector<double> w_bin;

    Stats angular_velocity_stats(angular_velocity_total);
    angular_velocity_stats.histogram(w_range, w_bin);

    norm = 1.0 / angular_velocity_total.size();
    for (int i = 0; i < w_bin.size(); ++i) {
      w_dist_file << i << "\t" << 0.5*(w_range[i] + w_range[i+1]) << "\t" << w_bin[i] * norm << "\n";
    }
    w_dist_file << "\n" << std::endl;



  }
}

SpinPumpingMonitor::~SpinPumpingMonitor() {
  w_dist_file.close();
  iz_dist_file.close();
  iz_mean_file.close();
}
