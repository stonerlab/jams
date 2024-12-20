// init_bloch_domain_wall.cc                                           -*-C++-*-

#include <jams/initializer/init_bloch_domain_wall.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <cmath>

void jams::InitBlochDomainWall::execute(const libconfig::Setting &settings) {
  double width = jams::config_required<double>(settings, "width");
  double center = jams::config_required<double>(settings, "center");

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r = globals::lattice->lattice_site_position_cart(i);
    // NOTE: The factor of pi here is an arbitrary convention. See the
    // documentation for jams::InitBlochDomainWall.
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 1.0 / std::cosh(kPi * (r[0]-center) / width);
    globals::s(i, 2) = std::tanh(kPi * (r[0]-center) / width);
  }
}

// ----------------------------- END-OF-FILE ----------------------------------
