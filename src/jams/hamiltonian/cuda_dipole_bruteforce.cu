#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "cuda_dipole_bruteforce.h"
#include "cuda_dipole_bruteforce_kernel.cuh"

CudaDipoleBruteforceHamiltonian::CudaDipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {
    Vec3 super_cell_dim = {0.0, 0.0, 0.0};

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice->size(n));
    }

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

    if (r_cutoff_ > lattice->max_interaction_radius()) {
      throw std::runtime_error(
          "r_cutoff is less than the maximum permitted interaction in the system"
          " (" + std::to_string(lattice->max_interaction_radius())  + ")");
    }

    auto v = pow3(lattice->parameter());
    dipole_prefactor_ = kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * v);

    bool super_cell_pbc[3];
    Mat<float,3,3> super_unit_cell;
    Mat<float,3,3> super_unit_cell_inv;

    for (int i = 0; i < 3; ++i) {
        super_cell_pbc[i] = ::lattice->is_periodic(i);
    }

    auto A = ::lattice->a() * double(::lattice->size(0));
    auto B = ::lattice->b() * double(::lattice->size(1));
    auto C = ::lattice->c() * double(::lattice->size(2));

    auto matrix_double = matrix_from_cols(A, B, C);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            super_unit_cell[i][j] = static_cast<float>(matrix_double[i][j]);
        }
    }

      super_unit_cell_inv = inverse(super_unit_cell);

    float r_cutoff_float = static_cast<float>(r_cutoff_);

    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_dipole_prefactor,    &dipole_prefactor_,       sizeof(double)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_r_cutoff,           &r_cutoff_float,       sizeof(float)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_super_cell_pbc,      super_cell_pbc,      3 * sizeof(bool)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_super_unit_cell,     &super_unit_cell[0][0],     9 * sizeof(float)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_super_unit_cell_inv, &super_unit_cell_inv[0][0], 9 * sizeof(float)));

    mus_float_.resize(globals::num_spins);
    for (auto i = 0; i < globals::num_spins; ++i) {
      mus_float_(i) = static_cast<float>(globals::mus(i));
    }

    r_float_.resize(globals::num_spins, 3);
    for (auto i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
          r_float_(i, j) = lattice->atom_position(i)[j];
        }
    }
}

// --------------------------------------------------------------------------

double CudaDipoleBruteforceHamiltonian::calculate_total_energy() {
    double e_total = 0.0;

    calculate_fields();
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += -0.5 * (  globals::s(i,0)*field_(i,0)
                           + globals::s(i,1)*field_(i,1)
                           + globals::s(i,2)*field_(i,2) );
    }

    return e_total;
}

double CudaDipoleBruteforceHamiltonian::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    const auto field = calculate_field(i);
    return -0.5 * dot(s_i, field);
}

double CudaDipoleBruteforceHamiltonian::calculate_energy(const int i) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i);
}

double CudaDipoleBruteforceHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    const auto field = calculate_field(i);
    double e_initial = -dot(spin_initial, field);
    double e_final = -dot(spin_final, field);
    return 0.5*(e_final - e_initial);
}

void CudaDipoleBruteforceHamiltonian::calculate_energies() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_energy(i);
    }
}

Vec3 CudaDipoleBruteforceHamiltonian::calculate_field(const int i) {
  using namespace globals;

  const auto neighbours = lattice->atom_neighbours(i, r_cutoff_);
  const double w0 = mus(i) * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));
  const Vec3 r_i = lattice->atom_position(i);
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3 field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbours) {
    const int j = neighbour.second;
    if (j == i) continue;

    const Vec3 s_j = {s(j,0), s(j,1), s(j,2)};
    const Vec3 r_ij =  neighbour.first - r_i;

    field += w0 * mus(j) * (3.0 * r_ij * dot(s_j, r_ij) - (norm(r_ij)*norm(r_ij)) * s_j) / pow5(norm(r_ij));
  }
  return field;
}

void CudaDipoleBruteforceHamiltonian::calculate_fields() {
    CudaStream stream;

    DipoleBruteforceKernel<<<(globals::num_spins + block_size - 1)/block_size, block_size, 0, stream.get() >>>
        (globals::s.device_data(), r_float_.device_data(), mus_float_.device_data(), globals::num_spins, field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
