// init_skyrmion.cc                                                    -*-C++-*-

#include <jams/initializer/init_skyrmion.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <cmath>

void jams::InitSkyrmion::execute(const libconfig::Setting &settings) {
  using namespace globals;

  double radius = jams::config_required<double>(settings, "radius");


  // Defaults to a Bloch skyrmion with a negative charge (core pointing
  // in -z direction within a +z FM)
  int Q = jams::config_optional<double>(settings, "charge", -1);
  double Qv = jams::config_optional<double>(settings, "vorticity", 1.0);
  double Qh = jams::config_optional<double>(settings, "helicity", 0.0);

  for (auto i = 0; i < num_spins; ++i) {
    auto r_i = ::lattice->displacement({0.0, 0.0, 0.0}, ::lattice->atom_position(i));
    double x = r_i[0];
    double y = r_i[1];
    double r = sqrt(x*x + y*y);

    // calculate orientation of skyrmion spin in perfect (T=0K) state

    // https://juspin.de/skyrmion-radius/
    // https://iopscience.iop.org/article/10.1088/1361-648X/ab5488
    double c = 5.0;
    double w = 5.0;
    double theta = asin(std::tanh(-(r+c)/(w/2))) + asin(std::tanh(-(r-c)/(w/2))) + kPi;
    double phi = std::atan2(y, x);


    globals::s(i,0) = sin(theta) * cos(Qv*phi + Qh);
    globals::s(i,1) = sin(theta) * sin(Qv*phi + Qh);
    globals::s(i,2) = cos(theta);

  }
}

// ----------------------------- END-OF-FILE ----------------------------------
