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
#include "jams/helpers/output.h"

#include "skyrmion.h"

SkyrmionMonitor::SkyrmionMonitor(const libconfig::Setting &settings)
: Monitor(settings),
 outfile(jams::output::full_path_filename("sky.tsv")){
  type_norms.resize(globals::lattice->num_materials(), 1.0);

  if (settings.exists("type_norms")) {
    for (int n = 0; n < globals::lattice->num_materials(); ++n) {
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
  for (double threshold : thresholds) {
    std::cout << "    " << threshold << "\n";
  }
  
  outfile.setf(std::ios::right);

  outfile << tsv_header();

  create_center_of_mass_mapping();
}

void SkyrmionMonitor::update(Solver * solver) {
    double x, y;

    const double x_size = globals::lattice->rmax()[0];
    const double y_size = globals::lattice->rmax()[1];

    outfile << std::setw(12) << std::scientific << solver->time();
    outfile << std::setw(16) << std::fixed << solver->physics()->temperature();

    for (double threshold : thresholds) {
      std::vector<Vec3 > r_com(globals::lattice->num_materials(), {0.0, 0.0, 0.0});
      calc_center_of_mass(r_com, threshold);

      int r_count[globals::lattice->num_materials()];
      double radius_gyration[globals::lattice->num_materials()];

      for (auto i = 0; i < globals::lattice->num_materials(); ++i) {
        r_count[i] = 0;
        radius_gyration[i] = 0.0;
      }

      for (auto i = 0; i < globals::num_spins; ++i) {
        auto type = globals::lattice->atom_material_id(i);
        if (globals::s(i, 2)*type_norms[type] > threshold) {
          x = globals::lattice->atom_position(i)[0] - r_com[type][0];
          x = x - nint(x / x_size) * x_size;  // min image convention
          y = globals::lattice->atom_position(i)[1] - r_com[type][1];
          y = y - nint(y / y_size) * y_size;  // min image convention
          radius_gyration[type] += x*x + y*y;
          r_count[type]++;
        }
      }

      for (auto n = 0; n < globals::lattice->num_materials(); ++n) {
        radius_gyration[n] = sqrt(radius_gyration[n]/static_cast<double>(r_count[n]));
      }

      for (auto n = 0; n < globals::lattice->num_materials(); ++n) {
        if (r_count[n] == 0) {
          for (auto i = 0; i < 5; ++i) {
            outfile << std::setw(16) << 0.0;
          }
        } else {
          for (auto i = 0; i < 3; ++i) {
            outfile << std::setw(16) << r_com[n][i]*globals::lattice->parameter();
          }
          outfile << std::setw(16) << radius_gyration[n]*globals::lattice->parameter() << std::setw(16) << (2.0/sqrt(2.0))*radius_gyration[n]*globals::lattice->parameter();
        }
      }
    }

    outfile << "\n";

}

void SkyrmionMonitor::create_center_of_mass_mapping() {
  // map the 2D x-y coordinate space onto 3D tubes for calculating the COM
  // in a periodic system
  // (see L. Bai and D. Breen, Journal of Graphics, GPU, and Game Tools 13, 53 (2008))

  tube_x.resize(globals::num_spins, {0.0, 0.0, 0.0});
  tube_y.resize(globals::num_spins, {0.0, 0.0, 0.0});

  for (int n = 0; n < globals::num_spins; ++n) {
    double i, j, i_max, j_max, r_i, r_j, theta_i, theta_j, x, y, z;

    i = globals::lattice->atom_position(n)[0];
    j = globals::lattice->atom_position(n)[1];

    i_max = globals::lattice->rmax()[0];
    j_max = globals::lattice->rmax()[1];

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
  // TODO: make the x and y PBC individually optional

  assert(tube_x.size() > 0);
  assert(tube_y.size() > 0);

  const int num_types = globals::lattice->num_materials();

  std::vector<Vec3 > tube_x_com(num_types, {0.0, 0.0, 0.0});
  std::vector<Vec3 > tube_y_com(num_types, {0.0, 0.0, 0.0});
  int r_count[num_types];

  for (auto type = 0; type < num_types; ++type) {
    r_count[type] = 0;
  }

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto type = globals::lattice->atom_material_id(i);
    if (globals::s(i, 2)*type_norms[type] > threshold) {
      tube_x_com[type] += tube_x[i];
      tube_y_com[type] += tube_y[i];
      r_count[type]++;
    }
  }

  for (auto type = 0; type < num_types; ++type) {
    r_com[type] /= static_cast<double>(r_count[type]);
  }

  for (auto type = 0; type < num_types; ++type) {
    double theta_i = atan2(-tube_x_com[type][2], -tube_x_com[type][0]) + kPi;
    double theta_j = atan2(-tube_y_com[type][2], -tube_y_com[type][1]) + kPi;

    r_com[type][0] = (theta_i*globals::lattice->rmax()[0]/(kTwoPi));
    r_com[type][1] = (theta_j*globals::lattice->rmax()[1]/(kTwoPi));
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

std::string SkyrmionMonitor::tsv_header() {
  std::stringstream ss;
  ss << std::setw(11) << "time\t";
  ss << std::setw(16) << "temperature\t";

  for (int i = 0; i < globals::lattice->num_materials(); ++i) {
    ss << std::setw(16) << globals::lattice->material_name(i) + ":r_avg_x\t";
    ss << std::setw(16) << globals::lattice->material_name(i) + ":r_avg_y\t";
    ss << std::setw(16) << globals::lattice->material_name(i) + ":r_avg_z\t";

    for (const auto threshold : thresholds) {
      ss << std::setw(16) << "R_g" << "[t" << threshold << "]\t";
      ss << std::setw(16) << "2R_g/sqrt(2)" << "[t" << threshold << "]\t";
    }
  }
  ss << "\n";
  return ss.str();
}


