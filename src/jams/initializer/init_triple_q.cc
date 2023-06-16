#include <jams/initializer/init_triple_q.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>


void jams::InitTripleQ::execute(const libconfig::Setting &settings) {
  Vec3 spin = jams::config_required<Vec3>(settings, "spin");


  double h = jams::config_required<double>(settings, "h");
  double k = jams::config_required<double>(settings, "k");
  double l = jams::config_required<double>(settings, "l");

  Vec3 K1 = globals::lattice->get_unitcell().a_inv();
  Vec3 K2 = globals::lattice->get_unitcell().b_inv();
  Vec3 K3 = globals::lattice->get_unitcell().c_inv();

  const std::string material = jams::config_optional<std::string>(settings, "material", "");


  for (auto i = 0; i < globals::num_spins; ++i) {
    if (material != "" && globals::lattice->atom_material_name(i) != material) {
      continue;
    }

    Vec3 r = {globals::positions(i, 0), globals::positions(i, 1), globals::positions(i, 2)};
    std::complex<double> phase = exp(kImagTwoPi * dot(h*K1 + k*K2 + l*K3, r));

    if (phase.imag() > 1e-5) {
      throw std::runtime_error("invalid triple Q parameters have given a complex phase");
    }

    Vec3 new_spin = phase.real() * spin;

    globals::s(i, 0) = new_spin[0];
    globals::s(i, 1) = new_spin[1];
    globals::s(i, 2) = new_spin[2];
  }
}