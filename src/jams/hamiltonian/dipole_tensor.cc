#include <cmath>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/dipole_tensor.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/timer.h"
#include <jams/lattice/interaction_neartree.h>

DipoleTensorHamiltonian::DipoleTensorHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {

  settings.lookupValue("r_cutoff", r_cutoff_);
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
  }

  jams::InteractionNearTree neartree(globals::lattice->get_supercell().a1(),
                                     globals::lattice->get_supercell().a2(),
                                     globals::lattice->get_supercell().a3(), globals::lattice->periodic_boundaries(), r_cutoff_, jams::defaults::lattice_tolerance);
  neartree.insert_sites(globals::lattice->lattice_site_positions_cart());

  int expected_neighbours = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    expected_neighbours += neartree.num_neighbours(
        globals::lattice->lattice_site_position_cart(i), r_cutoff_);
  }

  std::size_t max_memory_per_tensor = 9*(2*sizeof(int) + sizeof(double));

  std::cout << "  dipole dense tensor memory (not used) "
    << memory_in_natural_units(max_memory_per_tensor * pow2(globals::num_spins)) << std::endl;

  std::cout << "  dipole sparse matrix memory estimate (upper bound) "
    << memory_in_natural_units(max_memory_per_tensor * expected_neighbours) << std::endl;

  const double prefactor = kVacuumPermeabilityIU / (4 * kPi * pow(::globals::lattice->parameter(), 3));

  int num_neighbours = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    const Vec3 r_i = globals::lattice->lattice_site_position_cart(i);

    const auto neighbours = neartree.neighbours(r_i, r_cutoff_);
    for (const auto & neighbour : neighbours) {
      const int j = neighbour.second;
      assert(j >= 0 && j < globals::num_spins);
      if (j == i) continue;

      const auto r_ij =  neighbour.first - r_i;
      const auto r_abs = norm(r_ij);
      const auto r_hat = r_ij / r_abs;

      Mat3 dipole_tensor = kZeroMat3;
      for (auto m : {0, 1, 2}) {
        for (auto n : {0, 1, 2}) {
          dipole_tensor[m][n] +=
              (3.0 * r_hat[m] * r_hat[n] - kIdentityMat3[m][n]) * globals::mus(i) * globals::mus(j) /
              pow3(r_abs);
        }
      }
      num_neighbours++;
      insert_interaction_tensor(i, j, prefactor * dipole_tensor);
    }
  }

  Timer<> timer;
  finalize(jams::SparseMatrixSymmetryCheck::None);
  std::cout << "  build time " << timer.elapsed_time() << " seconds" << std::endl;

  std::cout << "  num_neighbours " << num_neighbours << "\n";
}