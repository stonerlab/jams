// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/smr.h"

SMRMonitor::SMRMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  smr(::lattice.num_materials()),
  outfile()
{
  using namespace globals;
  ::output.write("\ninitialising SMR monitor\n");

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_smr.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << std::setw(12) << "time" << "\t";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(12) << lattice.material_name(i) + ":smr" << "\t";
  }

  outfile << "\n";
}

void SMRMonitor::update(Solver * solver) {
  using namespace globals;

  // We assume the field is along the z direction
  // then the SMR is sin(theta)^2 - cos(theta)^2
  // theta = acos(Sz)
  // so SMR = sin(acos(x))^2 - cos(acos(x))^2
  //        = 1 - 2 Sz^2
    int i;

    smr.zero();

    for (i = 0; i < num_spins; ++i) {
      int type = lattice.atom_material(i);
      smr(type) += 1.0 - 2.0 * s(i, 2) * s(i, 2);
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
        smr(i) = smr(i)/static_cast<double>(lattice.num_of_material(i));
    }

    outfile << std::setw(12) << std::scientific << solver->time() << "\t";

    for (i = 0; i < lattice.num_materials(); ++i) {
      outfile << std::setw(12) << smr(i) << "\t";
    }

    outfile << std::endl;
}

SMRMonitor::~SMRMonitor() {
  outfile.close();
}
