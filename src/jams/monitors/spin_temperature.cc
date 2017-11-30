// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>

#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "spin_temperature.h"

#include "jblib/containers/array.h"

SpinTemperatureMonitor::SpinTemperatureMonitor(const libconfig::Setting &settings)
: Monitor(settings)
{
  using namespace globals;

  std::string name = seedname + "_T.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << std::setw(12) << "time" << "\t";
  outfile << std::setw(12) << "thermostat T" << "\t";
  outfile << std::setw(12) << "spin T" << "\t";

  outfile << "\n";
}

void SpinTemperatureMonitor::update(Solver * solver) {
  using namespace globals;

  auto sum_s_dot_h = 0.0;
  auto sum_s_cross_h = 0.0;
  
  for (auto i = 0; i < num_spins; ++i) {
    
    double sxh[3] = 
      {s(i, 1)*h(i, 2) - s(i, 2)*h(i, 1),
       s(i, 2)*h(i, 0) - s(i, 0)*h(i, 2),
       s(i, 0)*h(i, 1) - s(i, 1)*h(i, 0)};

    sum_s_cross_h += (sxh[0] * sxh[0] + sxh[1] * sxh[1] + sxh[2] * sxh[2]);

    sum_s_dot_h += (s(i,0) * h(i,0) + s(i,1) * h(i,1) + s(i,2) * h(i,2));
  }

  const auto spin_temperature = kBohrMagneton * sum_s_cross_h / (2.0 * kBoltzmann * sum_s_dot_h);

  outfile << std::setw(12) << std::scientific << solver->time() << "\t";
  outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";
  outfile << std::setw(12) << std::scientific << spin_temperature << "\n";
}

bool SpinTemperatureMonitor::is_converged() {
  return false;
}

SpinTemperatureMonitor::~SpinTemperatureMonitor() {
  outfile.close();
}
