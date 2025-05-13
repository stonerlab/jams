// init_bloch_domain_wall.cc                                           -*-C++-*-

#include <jams/initializer/init_bloch_domain_wall.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <cmath>

void jams::InitBlochDomainWall::execute(const libconfig::Setting &settings) {
  double width = jams::config_required<double>(settings, "width");
  double center = jams::config_required<double>(settings, "center");
  Vec3 normal = normalize(jams::config_optional<Vec3>(settings, "normal", {1, 0, 0}));
  Vec3 domain = normalize(jams::config_optional<Vec3>(settings, "domain", {0, 0, 1}));

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r = globals::lattice->lattice_site_position_cart(i);
    // NOTE: The factor of pi here is an arbitrary convention. See the
    // documentation for jams::InitBlochDomainWall.

    double x = dot(r, normal) - center;
    Vec3 m = {0, 1.0 / std::cosh(kPi * x / width), std::tanh(kPi * x / width)};

    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    auto rotation_matrix = rotation_matrix_between_vectors(domain, m);
    spin = rotation_matrix * spin;

    for (auto n = 0; n < 3; ++n) {
      globals::s(i, n) = spin[n];
    }
  }
}

// ----------------------------- END-OF-FILE ----------------------------------
