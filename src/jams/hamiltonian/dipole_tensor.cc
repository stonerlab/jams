#include <cmath>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/dipole_tensor.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"

using namespace std;

DipoleHamiltonianTensor::DipoleHamiltonianTensor(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {

  r_cutoff_ = settings["r_cutoff"];

  int expected_neighbours = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    expected_neighbours += lattice->num_neighbours(i, r_cutoff_);
  }

  std::size_t max_memory_per_tensor = 9*(2*sizeof(int) + sizeof(double));

  cout << "  dipole dense tensor memory (not used) "
    << memory_in_natural_units(max_memory_per_tensor * pow2(globals::num_spins)) << endl;

  cout << "  dipole sparse matrix memory estimate (upper bound) "
    << memory_in_natural_units(max_memory_per_tensor * expected_neighbours) << endl;

  const double prefactor = kVacuumPermeadbility * kBohrMagneton / (4 * kPi * pow(::lattice->parameter(), 3));

  int num_neighbours = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    const Vec3 r_i = lattice->atom_position(i);

    const auto neighbours = lattice->atom_neighbours(i, r_cutoff_);
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

  finalize(jams::SparseMatrixSymmetryCheck::None);
  cout << "  num_neighbours " << num_neighbours << "\n";
}