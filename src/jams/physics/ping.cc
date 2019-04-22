// Copyright 2014 Joseph Barker. All rights reserved.

#include "ping.h"

#include <libconfig.h++>

#include <cmath>

#include "jams/helpers/maths.h"
#include "jams/core/globals.h"

PingPhysics::PingPhysics(const libconfig::Setting &settings)
: Physics(settings) {

  double init_theta = 0.0, final_theta = 0.0, delta_theta = 0.0;
  double init_phi = 0.0, final_phi = 0.0, delta_phi = 0.0;

  bool theta_rotation_specified = false;
  bool phi_rotation_specified = false;

  if (settings.exists("theta")) {
    final_theta = settings["theta"];
    theta_rotation_specified = true;
  }

  if (settings.exists("phi")) {
    final_phi = settings["phi"];
    phi_rotation_specified = true;
  }

  Vec3 mag = {0,0,0};
  // find theta and phi of magnetisation
  for (int i = 0; i < globals::num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      mag[j] += globals::s(i, j) * globals::mus(i);
    }
  }

  init_theta = rad_to_deg(acos(mag[2] / abs(mag)));
  init_phi = rad_to_deg(atan2(mag[1], mag[0]));

  std::cout << "  initial angles (theta, phi) " << init_theta << " " << init_phi << "\n";
  std::cout << "  final angles (theta, phi) " << final_theta << " " << final_phi << "\n";

  if (theta_rotation_specified) {
    delta_theta = final_theta - init_theta;
  }

  if (phi_rotation_specified) {
    delta_phi = final_phi - init_phi;
  }

  std::cout << "  delta angles (theta, phi) " << delta_theta << " " << delta_phi << "\n";

  const double c_t = cos(deg_to_rad(delta_theta));
  const double c_p = cos(deg_to_rad(delta_phi));
  const double s_t = sin(deg_to_rad(delta_theta));
  const double s_p = sin(deg_to_rad(delta_phi));

  Mat3 rotation_matrix;
  Mat3 r_y;
  Mat3 r_z;

  // first index is row second index is col
  r_y[0][0] =  c_t;  r_y[0][1] =  0.0; r_y[0][2] =  s_t;
  r_y[1][0] =  0.0;  r_y[1][1] =  1.0; r_y[1][2] =  0.0;
  r_y[2][0] = -s_t;  r_y[2][1] =  0.0; r_y[2][2] =  c_t;

  r_z[0][0] =  c_p;  r_z[0][1] = -s_p;  r_z[0][2] =  0.0;
  r_z[1][0] =  s_p;  r_z[1][1] =  c_p;  r_z[1][2] =  0.0;
  r_z[2][0] =  0.0;  r_z[2][1] =  0.0;  r_z[2][2] =  1.0;

  rotation_matrix = r_y * r_z;

  Vec3 spin;
  for (int i = 0; i < globals::num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      spin[j] = globals::s(i, j);
    }

    spin = rotation_matrix * spin;

    for (int j = 0; j < 3; ++j) {
      globals::s(i, j) = spin[j];
    }
  }

  initialized = true;
}

PingPhysics::~PingPhysics() {
}

void PingPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
}
