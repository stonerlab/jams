// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "smr.h"

SMRMonitor::SMRMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file()
{
  std::cout << "\ninitialising SMR monitor\n";
  std::cout << "  assumes axes j->x, t->y, n->z\n";

  tsv_file.open(seedname + "_smr.tsv");
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void SMRMonitor::update(Solver * solver) {
  using namespace globals;

  std::vector<double> mtsq_para(lattice->num_materials(), 0.0);
  std::vector<double> mtsq_perp(lattice->num_materials(), 0.0);

  std::vector<double> mjmt_para(lattice->num_materials(), 0.0);
  std::vector<double> mjmt_perp(lattice->num_materials(), 0.0);

  std::vector<double> mn(lattice->num_materials(), 0.0);
  std::vector<int> material_count(lattice->num_materials(), 0);

  for (auto i = 0; i < num_spins; ++i) {
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
    if (material_count[i] > 0) {
      mtsq_para[i] = mtsq_para[i] / static_cast<double>(material_count[i]);
      mtsq_perp[i] = mtsq_perp[i] / static_cast<double>(material_count[i]);

      mjmt_para[i] = mjmt_para[i] / static_cast<double>(material_count[i]);
      mjmt_perp[i] = mjmt_perp[i] / static_cast<double>(material_count[i]);

      mn[i] = mn[i] / static_cast<double>(material_count[i]);
    }
  }

  tsv_file << std::setw(12) << std::scientific << solver->time() << "\t";

  for (int i = 0; i < lattice->num_materials(); ++i) {
    tsv_file << std::setw(12) << mtsq_para[i] << "\t" << mtsq_perp[i] << "\t";
    tsv_file << std::setw(12) << mjmt_para[i] << "\t" << mjmt_perp[i] << "\t";
    tsv_file << std::setw(12) << mn[i] << "\t";
  }

  tsv_file << std::endl;
}

std::string SMRMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (auto i = 0; i < lattice->num_materials(); ++i) {
    auto name = lattice->material_name(i);
    tsv_file << name + "_mtsq_para" << "\t";
    tsv_file << name + "_mtsq_perp" << "\t";
    tsv_file << name + "_mjmt_para" << "\t";
    tsv_file << name + "_mjmt_perp" << "\t";
    tsv_file << name + "_mn" << "\t";
  }

  ss << std::endl;

  return ss.str();
}
