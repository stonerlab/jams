// init_domain_wall.cc                                                          -*-C++-*-
#include <jams/initializer/init_domain_wall.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <string>
#include <cmath>

void jams::InitDomainWall::execute(const libconfig::Setting &settings) {
  using namespace globals;

  double width = jams::config_required<double>(settings, "width");
  double center = jams::config_required<double>(settings, "center");

  for (auto i = 0; i < num_spins; ++i) {
    auto r = ::lattice->atom_position(i);

    s(i, 0) = 0.0;
    s(i, 1) = 1.0 / std::cosh((r[0]-center) / width);
    s(i, 2) = std::tanh((r[0]-center) / width);

  }
}
