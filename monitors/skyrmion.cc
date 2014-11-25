// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <vector>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/maths.h"


#include "monitors/skyrmion.h"

SkyrmionMonitor::SkyrmionMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Skyrmion monitor...\n");

  is_equilibration_monitor_ = true;

  type_norms.resize(lattice.num_materials(), 1.0);

  if (settings.exists("type_norms")) {
    for (int n = 0; n < lattice.num_materials(); ++n) {
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

  ::output.write("  Sz thresholds:\n");
  for (int n = 0; n < thresholds.size(); ++n) {
    ::output.write("    %f\n", thresholds[n]);
  }

  std::string name = seedname + "_sky.dat";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << "#";
  outfile << std::setw(11) << "time";
  outfile << std::setw(16) << "temperature";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(16) <<  lattice.get_material_name(i) + " -> " + "r_avg(x,y,z)";
    for (int j = 0; j < thresholds.size(); ++j) {
      outfile << std::setw(16) << "R_g" << "[t" << thresholds[j] << "]";
      outfile << std::setw(16) << "2R_g/sqrt(2)" << "[t" << thresholds[j] << "]";
    }
  }
  outfile << "\n";

  create_center_of_mass_mapping();
}

void SkyrmionMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

    int i, n, type;

    outfile << std::setw(12) << std::scientific << time;
    outfile << std::setw(16) << std::fixed << temperature;


    for (int t = 0; t < thresholds.size(); ++t) {

      std::vector<jblib::Vec3<double> > r_com(lattice.num_materials(), jblib::Vec3<double>(0.0, 0.0, 0.0));
      calc_center_of_mass(r_com, thresholds[t]);

      int r_count[lattice.num_materials()];
      double radius_gyration[lattice.num_materials()];

      for (i = 0; i < lattice.num_materials(); ++i) {
        r_count[i] = 0;
        radius_gyration[i] = 0.0;
      }

      for (i = 0; i < num_spins; ++i) {
        type = lattice.get_material_number(i);
        if (s(i, 2)*type_norms[type] > thresholds[t]) {
          radius_gyration[type] += dot(lattice.lattice_positions_[i] - r_com[type], lattice.lattice_positions_[i] - r_com[type]);
          r_count[type]++;
        }
      }

      for (n = 0; n < lattice.num_materials(); ++n) {
        radius_gyration[n] = sqrt(radius_gyration[n]/static_cast<double>(r_count[n]));
      }

      for (n = 0; n < lattice.num_materials(); ++n) {
        if (r_count[n] == 0) {
          for (i = 0; i < 5; ++i) {
            outfile << std::setw(16) << 0.0;
          }
        } else {
          for (i = 0; i < 3; ++i) {
            outfile << std::setw(16) << r_com[n][i]*lattice.lattice_parameter_;
          }
          outfile << std::setw(16) << radius_gyration[n]*lattice.lattice_parameter_ << std::setw(16) << (2.0/sqrt(2.0))*radius_gyration[n]*lattice.lattice_parameter_;
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

  tube_x.resize(num_spins, jblib::Vec3<double>(0.0, 0.0, 0.0));
  tube_y.resize(num_spins, jblib::Vec3<double>(0.0, 0.0, 0.0));

  for (int n = 0; n < num_spins; ++n) {
    double i, j, i_max, j_max, r_i, r_j, theta_i, theta_j, x, y, z;

    i = lattice.lattice_positions_[n].x;
    j = lattice.lattice_positions_[n].y;

    i_max = lattice.rmax.x;
    j_max = lattice.rmax.y;

    r_i = i_max / (2.0 * pi);
    r_j = j_max / (2.0 * pi);

    theta_i = (i / i_max) * (2.0 * pi);
    theta_j = (j / j_max) * (2.0 * pi);

    x = r_i * cos(theta_i);
    y = j;
    z = r_i * sin(theta_i);

    tube_x[n] = jblib::Vec3<double>(x, y, z);

    x = i;
    y = r_j * cos(theta_j);
    z = r_j * sin(theta_j);

    tube_y[n] = jblib::Vec3<double>(x, y, z);
  }
}

void SkyrmionMonitor::calc_center_of_mass(std::vector<jblib::Vec3<double> > &r_com, const double &threshold) {
  using namespace globals;
  // TODO: make the x and y PBC individually optional

  assert(tube_x.size() > 0);
  assert(tube_y.size() > 0);

  const int num_types = lattice.num_materials();
  int i, type;
  double theta_i, theta_j;

  std::vector<jblib::Vec3<double> > tube_x_com(num_types, jblib::Vec3<double>(0.0, 0.0, 0.0));
  std::vector<jblib::Vec3<double> > tube_y_com(num_types, jblib::Vec3<double>(0.0, 0.0, 0.0));
  int r_count[num_types];

  for (type = 0; type < num_types; ++type) {
    r_count[type] = 0;
  }

  for (i = 0; i < num_spins; ++i) {
    type = lattice.get_material_number(i);
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
    theta_i = atan2(-tube_x_com[type].z, -tube_x_com[type].x) + pi;
    theta_j = atan2(-tube_y_com[type].z, -tube_y_com[type].y) + pi;

    r_com[type].x = (theta_i*lattice.rmax.x/(2.0*pi));
    r_com[type].y = (theta_j*lattice.rmax.y/(2.0*pi));
    r_com[type].z = 0.0;
  }

  // for (type = 0; type < num_types; ++type) {
  //   r_count[type] = 0;
  // }

  // for (i = 0; i < num_spins; ++i) {
  //   type = lattice.get_material_number(i);
  //   if (s(i, 2)*type_norms[type] > threshold) {
  //     r_com[type] += lattice.lattice_positions_[i];
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
