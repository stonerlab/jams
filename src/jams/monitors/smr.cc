// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "smr.h"

#include "jblib/containers/array.h"

SMRMonitor::SMRMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  outfile()
{
  using namespace globals;
  std::cout << "\ninitialising SMR monitor\n";
  std::cout << "  assumes axes j->x, t->y, n->z\n";

  std::string name = seedname + "_smr.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << std::setw(12) << "time" << "\t";

  for (int i = 0; i < lattice->num_materials(); ++i) {
    outfile << std::setw(12) << lattice->material_name(i) + ":mtsq_para" << "\t";
    outfile << std::setw(12) << lattice->material_name(i) + ":mtsq_perp" << "\t";

    outfile << std::setw(12) << lattice->material_name(i) + ":mjmt_para" << "\t";
    outfile << std::setw(12) << lattice->material_name(i) + ":mjmt_perp" << "\t";

    outfile << std::setw(12) << lattice->material_name(i) + ":mn" << "\t";
  }

  outfile << "\n";
}

void SMRMonitor::update(Solver * solver) {
  using namespace globals;

  std::vector<double> mtsq_para(lattice->num_materials(), 0.0);
  std::vector<double> mtsq_perp(lattice->num_materials(), 0.0);

  std::vector<double> mjmt_para(lattice->num_materials(), 0.0);
  std::vector<double> mjmt_perp(lattice->num_materials(), 0.0);

  std::vector<double> mn(lattice->num_materials(), 0.0);
  std::vector<int> material_count(lattice->num_materials(), 0);

  for (int i = 0; i < num_spins; ++i) {
    // Uses the WMI geometry from M. Althammer,Phys. Rev. B 87, 224401 (2013).
    // assuming axes:
    // j -> x
    // t -> y
    // n -> z
    int type = lattice->atom_material_id(i);
    mtsq_para[type] +=  s(i, 1) * s(i, 1);
    mtsq_perp[type] +=  s(i, 0) * s(i, 0);
    mjmt_para[type] +=  s(i, 0) * s(i, 1);
    mjmt_perp[type] += -s(i, 0) * s(i, 1);

    mn[type]   += s(i, 2);
    material_count[type]++;
  }

  for (int i = 0; i < lattice->num_materials(); ++i) {
      mtsq_para[i] = mtsq_para[i]/static_cast<double>(material_count[i]);
      mtsq_perp[i] = mtsq_perp[i]/static_cast<double>(material_count[i]);

      mjmt_para[i] = mjmt_para[i]/static_cast<double>(material_count[i]);
      mjmt_perp[i] = mjmt_perp[i]/static_cast<double>(material_count[i]);

      mn[i] = mn[i]/static_cast<double>(material_count[i]);
  }

  outfile << std::setw(12) << std::scientific << solver->time() << "\t";

  for (int i = 0; i < lattice->num_materials(); ++i) {
    outfile << std::setw(12) << mtsq_para[i] << "\t" << mtsq_perp[i] << "\t";
    outfile << std::setw(12) << mjmt_para[i] << "\t" << mjmt_perp[i] << "\t";
    outfile << std::setw(12) << mn[i] << "\t";
  }

  outfile << std::endl;
}


SMRMonitor::~SMRMonitor() {
  outfile.close();
}
