// init_bloch_domain_wall.cc                                           -*-C++-*-

#include <jams/initializer/init_bloch_domain_wall.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <cmath>

void jams::InitBlochDomainWall::execute(const libconfig::Setting &settings) {
  using namespace globals;

  double width = jams::config_required<double>(settings, "width");
  double center = jams::config_required<double>(settings, "center");

  for (auto i = 0; i < num_spins; ++i) {
    auto r = ::lattice->atom_position(i);
    // NOTE: The factor of pi here is an arbitrary convention. See the
    // documentation for jams::InitBlochDomainWall.
    s(i, 0) = 0.0;
    s(i, 1) = 1.0 / std::cosh(kPi * (r[0]-center) / width);
    s(i, 2) = std::tanh(kPi * (r[0]-center) / width);
  }
}

// ----------------------------- END-OF-FILE ----------------------------------
