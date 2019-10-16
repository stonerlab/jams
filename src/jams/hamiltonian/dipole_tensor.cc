#include <cmath>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/dipole_tensor.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"

using namespace std;

DipoleHamiltonianTensor::DipoleHamiltonianTensor(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {
  Vec3i L_max = {0, 0, 0};
  Vec3 super_cell_dim = {0.0, 0.0, 0.0};

  for (auto n : {0, 1, 2}) {
    super_cell_dim[n] = 0.5 * double(lattice->size(n));
  }

  r_cutoff_ = settings["r_cutoff"];

  for (auto n : {0, 1, 2}) {
    if (lattice->is_periodic(n)) {
      L_max[n] = ceil(r_cutoff_ / super_cell_dim[n]);
    }
  }

  cout << "  image vector max extent (fractional) " << L_max[0] << " " << L_max[1] << " " << L_max[2] << "\n";

  size_t row_memory = 3 * globals::num_spins * sizeof(int);
  size_t col_memory = square(3 * globals::num_spins) * sizeof(int);
  size_t val_memory = square(3 * globals::num_spins) * sizeof(double);

  cout << "  dipole tensor upper bound memory estimate "
    << static_cast<double>(row_memory + col_memory + val_memory) / kBytesToMegaBytes << "(MB)" << endl;

  const double prefactor = kVacuumPermeadbility * kBohrMagneton / (4 * kPi * pow(::lattice->parameter(), 3));

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < globals::num_spins; ++j) {
      // loop over periodic images of the simulation lattice
      // this means r_cutoff can be larger than the simulation cell
      Mat3 dipole_tensor = kZeroMat3;
      for (auto Lx = -L_max[0]; Lx < L_max[0] + 1; ++Lx) {
        for (auto Ly = -L_max[1]; Ly < L_max[1] + 1; ++Ly) {
          for (auto Lz = -L_max[2]; Lz < L_max[2] + 1; ++Lz) {
            Vec3i image_vector = {Lx, Ly, Lz};

            auto r_ij =
                lattice->generate_image_position(lattice->atom_position(j), image_vector) - lattice->atom_position(i);

            auto r_abs = norm(r_ij);

            // i can interact with i in another image of the simulation cell (just not the 0, 0, 0 image)
            // so detect based on r_abs rather than i == j
            if (definately_greater_than(r_abs, r_cutoff_, jams::defaults::lattice_tolerance) ||
                unlikely(approximately_zero(r_abs, jams::defaults::lattice_tolerance))) {
              continue;
            }

            auto r_hat = r_ij / r_abs;

            for (auto m : {0, 1, 2}) {
              for (auto n : {0, 1, 2}) {
                dipole_tensor[m][n] +=
                    (3 * r_hat[m] * r_hat[n] - kIdentityMat3[m][n]) * globals::mus(i) * globals::mus(j) /
                    pow(r_abs, 3);
              }
            }

          }
        }
      }
      insert_interaction_tensor(i, j, prefactor * dipole_tensor);

    }
  }

  finalize();
}