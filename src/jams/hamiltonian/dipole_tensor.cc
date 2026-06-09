#include <cmath>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/dipole_interaction.h"
#include "jams/hamiltonian/dipole_tensor.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/timer.h"
#include <jams/interface/config.h>
#include <jams/lattice/interaction_neartree.h>

DipoleTensorHamiltonian::DipoleTensorHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {

  r_cutoff_ = jams::config_required<jams::Real>(settings, "r_cutoff");
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
  }

  jams::InteractionNearTree<jams::Real> neartree(
    jams::array_cast<jams::Real>(globals::lattice->get_supercell().a1()),
    jams::array_cast<jams::Real>(globals::lattice->get_supercell().a2()),
    jams::array_cast<jams::Real>(globals::lattice->get_supercell().a3()),
    globals::lattice->periodic_boundaries(), r_cutoff_, jams::defaults::lattice_tolerance);

  std::vector<jams::Vec<jams::Real, 3>> positions;
  positions.reserve(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i)
  {
    positions.push_back(jams::array_cast<jams::Real>(jams::Vec<double, 3>{globals::positions(i,0), globals::positions(i,1), globals::positions(i,2)}));
  }
  neartree.insert_sites(positions);

  int expected_neighbours = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    expected_neighbours += neartree.num_neighbours(
        jams::Vec<jams::Real, 3>{globals::positions(i,0), globals::positions(i,1), globals::positions(i,2)}, r_cutoff_);
  }

  std::size_t max_memory_per_tensor = 9*(2*sizeof(int) + sizeof(jams::Real));

  std::cout << "  dipole dense tensor memory (not used) "
    << memory_in_natural_units(max_memory_per_tensor * pow2(globals::num_spins)) << std::endl;

  std::cout << "  dipole sparse matrix memory estimate (upper bound) "
    << memory_in_natural_units(max_memory_per_tensor * expected_neighbours) << std::endl;

  int num_neighbours = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    const jams::Vec<jams::Real, 3> r_i{globals::positions(i,0), globals::positions(i,1), globals::positions(i,2)};

    const auto neighbours = neartree.neighbours(r_i, r_cutoff_);
    for (const auto & neighbour : neighbours) {
      const int j = neighbour.second;
      assert(j >= 0 && j < globals::num_spins);
      if (j == i) continue;

      const auto r_ij =  neighbour.first - r_i;
      const auto dipole_tensor = jams::dipole::interaction_tensor<jams::Real>(
          jams::array_cast<double>(r_ij),
          globals::mus(i),
          globals::mus(j),
          globals::lattice->parameter());
      num_neighbours++;
      insert_interaction_tensor(i, j, dipole_tensor);
    }
  }

  Timer<> timer;
  finalize(jams::SparseMatrixSymmetryCheck::None);
  std::cout << "  build time " << timer.elapsed_time() << " seconds" << std::endl;

  std::cout << "  num_neighbours " << num_neighbours << "\n";
}

void DipoleTensorHamiltonian::add_energy_current_interactions(
    jams::EnergyCurrentInteractionSink& sink) const {
  jams::InteractionNearTree<jams::Real> neartree(
      jams::array_cast<jams::Real>(globals::lattice->get_supercell().a1()),
      jams::array_cast<jams::Real>(globals::lattice->get_supercell().a2()),
      jams::array_cast<jams::Real>(globals::lattice->get_supercell().a3()),
      globals::lattice->periodic_boundaries(),
      r_cutoff_,
      jams::defaults::lattice_tolerance);

  std::vector<jams::Vec<jams::Real, 3>> positions;
  positions.reserve(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    positions.push_back(jams::array_cast<jams::Real>(
        jams::Vec<double, 3>{globals::positions(i, 0),
                             globals::positions(i, 1),
                             globals::positions(i, 2)}));
  }
  neartree.insert_sites(positions);

  for (auto i = 0; i < globals::num_spins; ++i) {
    const jams::Vec<jams::Real, 3> r_i{
        globals::positions(i, 0),
        globals::positions(i, 1),
        globals::positions(i, 2)};

    const auto neighbours = neartree.neighbours(r_i, r_cutoff_);
    for (const auto& neighbour : neighbours) {
      const int j = neighbour.second;
      assert(j >= 0 && j < globals::num_spins);
      if (j == i) {
        continue;
      }

      const auto r_ji = neighbour.first - r_i;
      const auto dipole_tensor = jams::dipole::interaction_tensor(
          jams::array_cast<double>(r_ji),
          globals::mus(i),
          globals::mus(j),
          globals::lattice->parameter());
      sink.insert(i, j, jams::array_cast<double>(r_ji), dipole_tensor);
    }
  }
}
