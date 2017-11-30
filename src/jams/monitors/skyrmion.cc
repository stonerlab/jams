// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <vector>
#include <cassert>

#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/maths.h"
#include "skyrmion.h"

#include "jblib/containers/array.h"

SkyrmionMonitor::SkyrmionMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;

  type_norms.resize(lattice->num_materials(), 1.0);

  if (settings.exists("type_norms")) {
    for (int n = 0; n < lattice->num_materials(); ++n) {
      type_norms[n] = settings["type_norms"][n];
    }
  }

  if (settings.exists("thresholds")) {
    for (int n = 0; n < settings["thresholds"].getLength(); ++n) {
      thresholds.push_back(settings["thresholds"][n]);
    }
  } else {
    thresholds.push_back(0.0);
  }

  std::cout << "  Sz thresholds:\n";
  for (int n = 0; n < thresholds.size(); ++n) {
    std::cout << "    " << thresholds[n] << "\n";
  }

  std::string name = seedname + "_sky.dat";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << "#";
  outfile << std::setw(11) << "time";
  outfile << std::setw(16) << "temperature";

  for (int i = 0; i < lattice->num_materials(); ++i) {
    outfile << std::setw(16) <<  lattice->material_name(i) + " -> " + "r_avg(x,y,z)";
    for (int j = 0; j < thresholds.size(); ++j) {
      outfile << std::setw(16) << "R_g" << "[t" << thresholds[j] << "]";
      outfile << std::setw(16) << "2R_g/sqrt(2)" << "[t" << thresholds[j] << "]";
    }
  }
  outfile << "\n";

  create_center_of_mass_mapping();
}

void SkyrmionMonitor::update(Solver * solver) {
  using namespace globals;

    int i, n, type;
    double x, y;

    const double x_size = lattice->rmax()[0];
    const double y_size = lattice->rmax()[1];

    outfile << std::setw(12) << std::scientific << solver->time();
    outfile << std::setw(16) << std::fixed << solver->physics()->temperature();

    for (int t = 0; t < thresholds.size(); ++t) {

      std::vector<Vec3 > r_com(lattice->num_materials(), {0.0, 0.0, 0.0});
      calc_center_of_mass(r_com, thresholds[t]);

      int r_count[lattice->num_materials()];
      double radius_gyration[lattice->num_materials()];

      for (i = 0; i < lattice->num_materials(); ++i) {
        r_count[i] = 0;
        radius_gyration[i] = 0.0;
      }

      for (i = 0; i < num_spins; ++i) {
        type = lattice->atom_material(i);
        if (s(i, 2)*type_norms[type] > thresholds[t]) {
          x = lattice->atom_position(i)[0] - r_com[type][0];
          x = x - nint(x / x_size) * x_size;  // min image convention
          y = lattice->atom_position(i)[1] - r_com[type][1];
          y = y - nint(y / y_size) * y_size;  // min image convention
          radius_gyration[type] += x*x + y*y;
          r_count[type]++;
        }
      }

      for (n = 0; n < lattice->num_materials(); ++n) {
        radius_gyration[n] = sqrt(radius_gyration[n]/static_cast<double>(r_count[n]));
      }

      for (n = 0; n < lattice->num_materials(); ++n) {
        if (r_count[n] == 0) {
          for (i = 0; i < 5; ++i) {
            outfile << std::setw(16) << 0.0;
          }
        } else {
          for (i = 0; i < 3; ++i) {
            outfile << std::setw(16) << r_com[n][i]*lattice->parameter();
          }
          outfile << std::setw(16) << radius_gyration[n]*lattice->parameter() << std::setw(16) << (2.0/sqrt(2.0))*radius_gyration[n]*lattice->parameter();
        }
      }
    }

    outfile << "\n";

}

void SkyrmionMonitor::create_center_of_mass_mapping() {
  using namespace globals;

  // map the 2D x-y coordinate space onto 3D tubes for calculating the COM
  // in a periodic system
  // (see L. Bai and D. Breen, Journal of Graphics, GPU, and Game Tools 13, 53 (2008))

  tube_x.resize(num_spins, {0.0, 0.0, 0.0});
  tube_y.resize(num_spins, {0.0, 0.0, 0.0});

  for (int n = 0; n < num_spins; ++n) {
    double i, j, i_max, j_max, r_i, r_j, theta_i, theta_j, x, y, z;

    i = lattice->atom_position(n)[0];
    j = lattice->atom_position(n)[1];

    i_max = lattice->rmax()[0];
    j_max = lattice->rmax()[1];

    r_i = i_max / (kTwoPi);
    r_j = j_max / (kTwoPi);

    theta_i = (i / i_max) * (kTwoPi);
    theta_j = (j / j_max) * (kTwoPi);

    x = r_i * cos(theta_i);
    y = j;
    z = r_i * sin(theta_i);

    tube_x[n] = {x, y, z};

    x = i;
    y = r_j * cos(theta_j);
    z = r_j * sin(theta_j);

    tube_y[n] = {x, y, z};
  }
}

void SkyrmionMonitor::calc_center_of_mass(std::vector<Vec3 > &r_com, const double &threshold) {
  using namespace globals;
  // TODO: make the x and y PBC individually optional

  assert(tube_x.size() > 0);
  assert(tube_y.size() > 0);

  const int num_types = lattice->num_materials();
  int i, type;
  double theta_i, theta_j;

  std::vector<Vec3 > tube_x_com(num_types, {0.0, 0.0, 0.0});
  std::vector<Vec3 > tube_y_com(num_types, {0.0, 0.0, 0.0});
  int r_count[num_types];

  for (type = 0; type < num_types; ++type) {
    r_count[type] = 0;
  }

  for (i = 0; i < num_spins; ++i) {
    type = lattice->atom_material(i);
    if (s(i, 2)*type_norms[type] > threshold) {
      tube_x_com[type] += tube_x[i];
      tube_y_com[type] += tube_y[i];
      r_count[type]++;
    }
  }

  for (type = 0; type < num_types; ++type) {
    r_com[type] /= static_cast<double>(r_count[type]);
  }

  for (type = 0; type < num_types; ++type) {
    theta_i = atan2(-tube_x_com[type][2], -tube_x_com[type][0]) + kPi;
    theta_j = atan2(-tube_y_com[type][2], -tube_y_com[type][1]) + kPi;

    r_com[type][0] = (theta_i*lattice->rmax()[0]/(kTwoPi));
    r_com[type][1] = (theta_j*lattice->rmax()[1]/(kTwoPi));
    r_com[type][2] = 0.0;
  }

  // for (type = 0; type < num_types; ++type) {
  //   r_count[type] = 0;
  // }

  // for (i = 0; i < num_spins; ++i) {
  //   type = lattice->atom_material_id(i);
  //   if (s(i, 2)*type_norms[type] > threshold) {
  //     r_com[type] += lattice->lattice_positions_[i];
  //     r_count[type]++;
  //   }
  // }

  // for (type = 0; type < num_types; ++type) {
  //   r_com[type] /= static_cast<double>(r_count[type]);
  // }

}


SkyrmionMonitor::~SkyrmionMonitor() {
  outfile.close();
}
