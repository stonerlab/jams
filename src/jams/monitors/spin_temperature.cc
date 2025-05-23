// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>

#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/containers/vec3.h"
#include "jams/helpers/output.h"
#include "spin_temperature.h"

SpinTemperatureMonitor::SpinTemperatureMonitor(const libconfig::Setting &settings)
: Monitor(settings),
tsv_file(jams::output::full_path_filename("spin_T.tsv"))
{
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void SpinTemperatureMonitor::update(Solver& solver) {
  double sum_s_dot_h = 0.0;
  double sum_s_cross_h = 0.0;

  #if HAS_OMP
  #pragma omp parallel for reduction(+:sum_s_cross_h, sum_s_dot_h)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    const Vec3 spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    const Vec3 field = {globals::h(i,0), globals::h(i,1), globals::h(i,2)};

    sum_s_cross_h += norm_squared(cross(spin, field));
    sum_s_dot_h += dot(spin, field);
  }

  const auto spin_temperature = sum_s_cross_h / (2.0 * kBoltzmannIU * sum_s_dot_h);

  tsv_file.width(12);
  tsv_file << std::scientific << solver.time() << "\t";
  tsv_file << std::fixed << solver.physics()->temperature() << "\t";
  tsv_file << std::scientific << spin_temperature << "\n";
}

std::string SpinTemperatureMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  ss << "thermostat_T" << "\t";
  ss << "spin_T" << "\t";
  ss << std::endl;

  return ss.str();
}
