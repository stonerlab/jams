// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/output.h"

#include "field.h"

FieldMonitor::FieldMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("fld.tsv")){
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void FieldMonitor::update(Solver * solver) {

  std::vector<Vec3> total_field;
  for (auto &hamiltonian : solver->hamiltonians()) {
    hamiltonian->calculate_fields(solver->time());

    Vec3 field = {0.0, 0.0, 0.0};
    for (auto i = 0; i < globals::num_spins; ++i) {
      field += Vec3{hamiltonian->field(i, 0), hamiltonian->field(i, 1), hamiltonian->field(i, 2)} / globals::mus(i);
    }

    total_field.push_back(field / static_cast<double>(globals::num_spins));
  }

  tsv_file.width(12);

  tsv_file << std::scientific << solver->time() << "\t";

  for (const auto& field : total_field) {
    for (auto n = 0; n < 3; ++n) {
      tsv_file << std::scientific << std::setprecision(8) << field[n] << "\t";
    }
  }

  tsv_file << std::endl;
}

std::string FieldMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (auto &hamiltonian : solver->hamiltonians()) {
    ss << hamiltonian->name() << "_hx\t";
    ss << hamiltonian->name() << "_hy\t";
    ss << hamiltonian->name() << "_hz\t";
  }

  ss << std::endl;

  return ss.str();
}
