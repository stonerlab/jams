// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <cassert>
#include <iostream>

#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/output.h"

#include "skyrmion.h"

SkyrmionMonitor::SkyrmionMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
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
  
  tsv_ = make_tsv_writer();

  create_center_of_mass_mapping();
}

void SkyrmionMonitor::update(Solver& solver) {
    double x, y;
    const auto spins = globals::s.host_view();

    const double x_size = globals::lattice->rmax()[0];
    const double y_size = globals::lattice->rmax()[1];

    std::vector<double> values;
    values.reserve(tsv_.num_cols());
    solver.append_monitor_coordinates(values);
    values.push_back(solver.physics()->temperature());

    for (double threshold : thresholds) {
      std::vector<jams::Vec<double, 3> > r_com(globals::lattice->num_materials(), {0.0, 0.0, 0.0});
      calc_center_of_mass(r_com, threshold);

      std::vector<int> r_count(globals::lattice->num_materials(), 0);
      std::vector<double> radius_gyration(globals::lattice->num_materials(), 0.0);

      for (auto i = 0; i < globals::num_spins; ++i) {
        auto type = globals::lattice->lattice_site_material_id(i);
        if (spins(i, 2)*type_norms[type] > threshold) {
          x = globals::lattice->lattice_site_position_cart(i)[0] - r_com[type][0];
          x = x - nint(x / x_size) * x_size;  // min image convention
          y = globals::lattice->lattice_site_position_cart(i)[1] - r_com[type][1];
          y = y - nint(y / y_size) * y_size;  // min image convention
          radius_gyration[type] += x*x + y*y;
          r_count[type]++;
        }
      }

      for (auto n = 0; n < globals::lattice->num_materials(); ++n) {
        if (r_count[n] > 0) {
          radius_gyration[n] = sqrt(radius_gyration[n] / static_cast<double>(r_count[n]));
          for (auto i = 0; i < 3; ++i) {
            values.push_back(r_com[n][i] * globals::lattice->parameter());
          }
          values.push_back(radius_gyration[n] * globals::lattice->parameter());
          values.push_back((2.0 / sqrt(2.0)) * radius_gyration[n] * globals::lattice->parameter());
        } else {
          for (auto i = 0; i < 5; ++i) {
            values.push_back(0.0);
          }
        }
      }
    }

    tsv_.write_row(values);
}

void SkyrmionMonitor::create_center_of_mass_mapping() {
  // map the 2D x-y coordinate space onto 3D tubes for calculating the COM
  // in a periodic system
  // (see L. Bai and D. Breen, Journal of Graphics, GPU, and Game Tools 13, 53 (2008))

  tube_x.resize(globals::num_spins, {0.0, 0.0, 0.0});
  tube_y.resize(globals::num_spins, {0.0, 0.0, 0.0});

  for (int n = 0; n < globals::num_spins; ++n) {
    double i, j, i_max, j_max, r_i, r_j, theta_i, theta_j, x, y, z;

    i = globals::lattice->lattice_site_position_cart(n)[0];
    j = globals::lattice->lattice_site_position_cart(n)[1];

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

void SkyrmionMonitor::calc_center_of_mass(std::vector<jams::Vec<double, 3> > &r_com, const double &threshold) {
  // TODO: make the x and y PBC individually optional

  assert(tube_x.size() > 0);
  assert(tube_y.size() > 0);

  const int num_types = globals::lattice->num_materials();
  const auto spins = globals::s.host_view();

  std::vector<jams::Vec<double, 3> > tube_x_com(num_types, {0.0, 0.0, 0.0});
  std::vector<jams::Vec<double, 3> > tube_y_com(num_types, {0.0, 0.0, 0.0});
  std::vector<int> r_count(num_types, 0);

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto type = globals::lattice->lattice_site_material_id(i);
    if (spins(i, 2)*type_norms[type] > threshold) {
      tube_x_com[type] += tube_x[i];
      tube_y_com[type] += tube_y[i];
      r_count[type]++;
    }
  }

  for (auto type = 0; type < num_types; ++type) {
    if (r_count[type] == 0) {
      continue;
    }
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
  //   type = lattice->lattice_site_material_id(i);
  //   if (s(i, 2)*type_norms[type] > threshold) {
  //     r_com[type] += lattice->lattice_positions_[i];
  //     r_count[type]++;
  //   }
  // }

  // for (type = 0; type < num_types; ++type) {
  //   r_com[type] /= static_cast<double>(r_count[type]);
  // }

}


SkyrmionMonitor::~SkyrmionMonitor() = default;

jams::output::TsvWriter SkyrmionMonitor::make_tsv_writer() const {
  auto cols = globals::solver->monitor_coordinate_columns();
  cols.push_back({"temperature", "K", jams::output::ColFmt::Fixed});

  for (const auto threshold : thresholds) {
    std::ostringstream threshold_label;
    threshold_label << "t" << threshold;
    for (int i = 0; i < globals::lattice->num_materials(); ++i) {
      const auto prefix = threshold_label.str() + "_" + globals::lattice->material_name(i);
      cols.push_back({prefix + "_r_avg_x", "m"});
      cols.push_back({prefix + "_r_avg_y", "m"});
      cols.push_back({prefix + "_r_avg_z", "m"});
      cols.push_back({prefix + "_R_g", "m"});
      cols.push_back({prefix + "_2R_g_sqrt2", "m"});
    }
  }

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols));
}
