// Copyright 2014 Joseph Barker. All rights reserved.

#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

#include "monitors/spin_pumping.h"

#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/maths.h"
#include "core/stats.h"
#include "core/hamiltonian.h"
#include "core/thermostat.h"

SpinPumpingMonitor::SpinPumpingMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  ::output.write("Initialising Energy Distribution monitor\n");


  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
    ::output.write("  convergence tolerance: %f\n", convergence_tolerance_);
  }

  std::string name = "_iz_dist.tsv";
  name = seedname+name;
  iz_dist_file.open(name.c_str());
  iz_dist_file.setf(std::ios::right);
  iz_dist_file << std::setw(12) << "bin" << "\t";
  iz_dist_file << std::setw(12) << "Iz_g_real" << "\t";
  iz_dist_file << std::setw(12) << "count" << std::endl;


  name = "_iz_mean.tsv";
  name = seedname+name;
  iz_mean_file.open(name.c_str());
  iz_mean_file.setf(std::ios::right);
  iz_mean_file << std::setw(12) << "time" << "\t";
  iz_mean_file << std::setw(12) << "Iz_g_real" << "\t";
  iz_mean_file << std::setw(12) << "Iz_g_imag" << "\t";

  if (convergence_is_on_) {
    iz_mean_file << std::setw(12) << "geweke";
  }

  iz_mean_file << std::endl;

  name = "_w_dist.tsv";
  name = seedname+name;
  w_dist_file.open(name.c_str());
  w_dist_file << std::setw(12) << "bin" << "\t";
  w_dist_file << std::setw(12) << "w_freq" << "\t";
  w_dist_file << std::setw(12) << "count" << std::endl;

}

void SpinPumpingMonitor::update(Solver * solver) {
  using namespace globals;
  using std::abs;

    std::vector<double> energy_total(num_spins, 0.0);
    std::vector<double> pumping_total(num_spins, 0.0);
    std::vector<double> imaginary_total(num_spins, 0.0);
    std::vector<double> noise_total(num_spins, 0.0);
    std::vector<double> angular_velocity_total(num_spins, 0.0);

    Stats pumping_abs_pos;
    Stats pumping_abs_neg;

    for (int i = 0; i < num_spins; ++i) {
      // double vt[3]= { 0.0,
      //                 0.0,
      //                 s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0) };

      pumping_total[i] = ((s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0)));

      noise_total[i] = ((s(i, 0)*solver->thermostat()->field(i, 1) - s(i, 1)*solver->thermostat()->field(i, 0)));

      if (pumping_total[i] > 0.0) {
        pumping_abs_pos.add(abs(pumping_total[i]));
      } else {
        pumping_abs_neg.add(abs(pumping_total[i]));
      }

      imaginary_total[i] = ds_dt(i, 2);
      angular_velocity_total[i] = ((s(i, 0)*ds_dt(i, 1) - s(i, 1)*ds_dt(i, 0))) / (s(i, 0)*s(i, 0) + s(i, 1)*s(i, 1));
    }

    // dt_sim = dt * gamma_e
    // so 1/dt_sm -> 1/dt means x gamma_e

    std::vector<double> range, range_pos, range_neg;
    std::vector<double> bin, bin_pos, bin_neg;


    Stats noise(noise_total);
    Stats pumping(pumping_total);
    Stats imaginary(imaginary_total);

    double p_min_max = std::max(abs(pumping.min()), abs(pumping.max()));


    pumping.histogram(range, bin, -p_min_max, p_min_max);

    pumping_abs_pos.histogram(range_pos, bin_pos, -p_min_max, p_min_max, bin.size());
    pumping_abs_neg.histogram(range_neg, bin_neg, -p_min_max, p_min_max, bin.size());

    double norm = 1.0 / pumping_total.size();
    for (int i = 0; i < bin.size(); ++i) {
      // output in rad / s^-1 T^-1
      iz_dist_file << std::setw(12) << i << "\t";
      iz_dist_file << std::setw(12) << std::scientific << 0.5*(range[i] + range[i+1]) * kGyromagneticRatio << "\t";
      iz_dist_file << std::setw(12) << std::scientific << bin[i] * norm << "\t";
      iz_dist_file << std::setw(12) << std::scientific << (bin_pos[i] / pumping.size()) << "\t";
      iz_dist_file << std::setw(12) << std::scientific << (bin_neg[i] / pumping.size()) << "\n";
      // iz_dist_file << std::setw(12) << std::scientific << (bin_pos[i]-bin_neg[i]) * norm << "\n";
    }
    iz_dist_file << "\n" << std::endl;

    // output in rad / s^-1 T^-1
    iz_mean_file << std::setw(12) << std::scientific << solver->time() << "\t";
    iz_mean_file << std::setw(12) << std::scientific << pumping.mean() * kGyromagneticRatio << "\t";
    iz_mean_file << std::setw(12) << std::scientific << imaginary.mean() * kGyromagneticRatio << "\t";
    iz_mean_file << std::setw(12) << std::scientific << noise.mean() * kGyromagneticRatio << "\t";


    if (convergence_is_on_) {
      convergence_stats_.add(pumping.mean());
      convergence_geweke_diagnostic_ = convergence_stats_.geweke();
      iz_mean_file << std::setw(12) << convergence_geweke_diagnostic_;
    }

    iz_mean_file << std::endl;

    Stats angular_velocity_stats(angular_velocity_total);
    angular_velocity_stats.histogram(range, bin);

    norm = 1.0 / angular_velocity_total.size();
    for (int i = 0; i < bin.size(); ++i) {
      // output in Hz
      w_dist_file << std::setw(12) << i << "\t";
      w_dist_file << std::setw(12) << std::scientific << 0.5*(range[i] + range[i+1]) * kGyromagneticRatio << "\t";
      w_dist_file << std::setw(12) << std::scientific << bin[i] * norm << "\n";
    }
    w_dist_file << "\n" << std::endl;

}

bool SpinPumpingMonitor::is_converged() {
  return ((std::abs(convergence_geweke_diagnostic_) < convergence_tolerance_) && convergence_is_on_);
}

SpinPumpingMonitor::~SpinPumpingMonitor() {
  w_dist_file.close();
  iz_dist_file.close();
  iz_mean_file.close();
}
