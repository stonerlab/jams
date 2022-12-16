#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/lattice/minimum_image.h"

#include "cuda_dipole_bruteforce.h"
#include "cuda_dipole_bruteforce_kernel.cuh"

CudaDipoleBruteforceHamiltonian::CudaDipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {
    Vec3 super_cell_dim = {0.0, 0.0, 0.0};

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(globals::lattice->size(n));
    }

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

    if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
      throw std::runtime_error(
          "r_cutoff is less than the maximum permitted interaction in the system"
          " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
    }

    auto v = pow3(globals::lattice->parameter());
    dipole_prefactor_ = kVacuumPermeabilityIU / (4.0 * kPi * v);

    bool super_cell_pbc[3];
    Mat<float,3,3> super_unit_cell;
    Mat<float,3,3> super_unit_cell_inv;

    for (int i = 0; i < 3; ++i) {
        super_cell_pbc[i] = ::globals::lattice->is_periodic(i);
    }

    auto A = ::globals::lattice->a() * double(::globals::lattice->size(0));
    auto B = ::globals::lattice->b() * double(::globals::lattice->size(1));
    auto C = ::globals::lattice->c() * double(::globals::lattice->size(2));

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
          r_float_(i, j) = globals::lattice->atom_position(i)[j];
        }
    }
}

// --------------------------------------------------------------------------

double CudaDipoleBruteforceHamiltonian::calculate_total_energy(double time) {
    double e_total = 0.0;

    calculate_fields(time);
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += -0.5 * (  globals::s(i,0)*field_(i,0)
                           + globals::s(i,1)*field_(i,1)
                           + globals::s(i,2)*field_(i,2) );
    }

    return e_total;
}

double CudaDipoleBruteforceHamiltonian::calculate_one_spin_energy(const int i, const Vec3 &s_i, double time) {
    const auto field = calculate_field(i, time);
    return -0.5 * dot(s_i, field);
}

double CudaDipoleBruteforceHamiltonian::calculate_energy(const int i, double time) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i, time);
}

double CudaDipoleBruteforceHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) {
    const auto field = calculate_field(i, time);
    double e_initial = -dot(spin_initial, field);
    double e_final = -dot(spin_final, field);
    return 0.5*(e_final - e_initial);
}

void CudaDipoleBruteforceHamiltonian::calculate_energies(double time) {
    for (auto i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_energy(i, time);
    }
}

Vec3 CudaDipoleBruteforceHamiltonian::calculate_field(const int i, double time) {
  using namespace std::placeholders;


  // We will use Smith's algorithm for the minimum image convention below which is only valid for
  // displacements less than the inradius of the cell. Our r_cutoff_ is checked at runtime in the
  // constructor for this condition which allows us to turn off the safety check in Smith's algorithm
  // (an optimisation). We assert the condition here again for safety.
  assert(r_cutoff_ < globals::lattice->max_interaction_radius());

  auto displacement = [](const int i, const int j)->Vec3 {
      return jams::minimum_image_smith_method(
          globals::lattice->get_supercell().matrix(),
          globals::lattice->get_supercell().inverse_matrix(),
          globals::lattice->get_supercell().periodic(),
          globals::lattice->atom_position(i),
          globals::lattice->atom_position(j));
  };

  const auto r_cut_squared = r_cutoff_ * r_cutoff_;
  const double w0 = globals::mus(i) * kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter()));

  double hx = 0, hy = 0, hz = 0;
#if HAS_OMP
#pragma omp parallel for reduction(+:hx, hy, hz)
#endif
  for (auto j = 0; j < globals::num_spins; ++j) {
    if (j == i) continue;

    const Vec3 s_j = {globals::s(j,0), globals::s(j,1), globals::s(j,2)};

    Vec3 r_ij = displacement(i, j);

    const auto r_abs_sq = norm_squared(r_ij);

    if (definately_greater_than(r_abs_sq, r_cut_squared, jams::defaults::lattice_tolerance*jams::defaults::lattice_tolerance)) continue;

    hx += w0 * globals::mus(j) * (3.0 * r_ij[0] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[0]) / pow5(norm(r_ij));
    hy += w0 * globals::mus(j) * (3.0 * r_ij[1] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[1]) / pow5(norm(r_ij));;
    hz += w0 * globals::mus(j) * (3.0 * r_ij[2] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[2]) / pow5(norm(r_ij));;
  }

  return {hx, hy, hz};
}

void CudaDipoleBruteforceHamiltonian::calculate_fields(double time) {
    CudaStream stream;

    DipoleBruteforceKernel<<<(globals::num_spins + block_size - 1)/block_size, block_size, 0, stream.get() >>>
        (globals::s.device_data(), r_float_.device_data(), mus_float_.device_data(), globals::num_spins, field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
