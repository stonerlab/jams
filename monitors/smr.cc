// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/smr.h"

SMRMonitor::SMRMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  outfile()
{
  using namespace globals;
  ::output.write("\ninitialising SMR monitor\n");
  ::output.write("  assumes axes j->x, t->y, n->z");

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_smr.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << std::setw(12) << "time" << "\t";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(12) << lattice.material_name(i) + ":mj" << "\t";
    outfile << std::setw(12) << lattice.material_name(i) + ":mt" << "\t";
    outfile << std::setw(12) << lattice.material_name(i) + ":mn" << "\t";
  }

  outfile << "\n";
}

void SMRMonitor::update(Solver * solver) {
  using namespace globals;

  std::vector<double> mj(lattice.num_materials(), 0.0);
  std::vector<double> mt(lattice.num_materials(), 0.0);
  std::vector<double> mn(lattice.num_materials(), 0.0);

  for (int i = 0; i < num_spins; ++i) {
    // Uses the WMI geometry from M. Althammer,Phys. Rev. B 87, 224401 (2013).
    // assuming axes:
    // j -> x
    // t -> y
    // n -> z
    int type = lattice.atom_material(i);
    mj[type] += s(i, 0);
    mt[type] += s(i, 1);
    mn[type] += s(i, 2);
  }

  for (int i = 0; i < lattice.num_materials(); ++i) {
      mj[i] = mj[i]/static_cast<double>(lattice.num_of_material(i));
      mt[i] = mt[i]/static_cast<double>(lattice.num_of_material(i));
      mn[i] = mn[i]/static_cast<double>(lattice.num_of_material(i));
  }

  outfile << std::setw(12) << std::scientific << solver->time() << "\t";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(12) << mj[i] << "\t" << mt[i] << "\t" << mn[i] << "\t";
  }

  outfile << std::endl;
}


SMRMonitor::~SMRMonitor() {
  outfile.close();
}
