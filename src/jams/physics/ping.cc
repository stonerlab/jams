// Copyright 2014 Joseph Barker. All rights reserved.

#include <jams/physics/ping.h>

#include <libconfig.h++>

#include <cmath>

#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/helpers/spinops.h"

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

  // find theta and phi of magnetisation
  Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus);

  init_theta = rad_to_deg(acos(mag[2] / norm(mag)));
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

  jams::rotate_spins(globals::s, rotation_matrix);

  initialized = true;
}

PingPhysics::~PingPhysics() {
}

void PingPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
}
