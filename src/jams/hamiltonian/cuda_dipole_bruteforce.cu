#include <jams/hamiltonian/cuda_dipole_bruteforce.h>
#include <jams/hamiltonian/cuda_dipole_bruteforce_kernel.cuh>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/helpers/consts.h>
#include <jams/lattice/minimum_image.h>

CudaDipoleBruteforceHamiltonian::CudaDipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {
    Vec3R super_cell_dim{0, 0, 0};

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = jams::Real(globals::lattice->size(n)) / 2;
    }

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

    if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
      throw std::runtime_error(
          "r_cutoff is less than the maximum permitted interaction in the system"
          " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
    }

    auto v = pow3(globals::lattice->parameter());
    dipole_prefactor_ = static_cast<jams::Real>(kVacuumPermeabilityIU / (4.0 * kPi * v));

    bool super_cell_pbc[3];
    Mat<jams::Real,3,3> super_unit_cell;
    Mat<jams::Real,3,3> super_unit_cell_inv;

    for (int i = 0; i < 3; ++i) {
        super_cell_pbc[i] = ::globals::lattice->is_periodic(i);
    }

    auto A = ::globals::lattice->a1() * double(::globals::lattice->size(0));
    auto B = ::globals::lattice->a2() * double(::globals::lattice->size(1));
    auto C = ::globals::lattice->a3() * double(::globals::lattice->size(2));

    auto matrix_double = matrix_from_cols(A, B, C);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            super_unit_cell[i][j] = static_cast<jams::Real>(matrix_double[i][j]);
        }
    }

      super_unit_cell_inv = inverse(super_unit_cell);

    const jams::Real r_cutoff_device = static_cast<jams::Real>(r_cutoff_);

    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_dipole_prefactor, &dipole_prefactor_, sizeof(dipole_prefactor_)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_r_cutoff, &r_cutoff_device, sizeof(r_cutoff_device)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_super_cell_pbc, super_cell_pbc, sizeof(super_cell_pbc)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_super_unit_cell, super_unit_cell.data(), sizeof(super_unit_cell)));
    CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_super_unit_cell_inv, super_unit_cell_inv.data(), sizeof(super_unit_cell_inv)));

    mus_float_.resize(globals::num_spins);
    for (auto i = 0; i < globals::num_spins; ++i) {
      mus_float_(i) = static_cast<float>(globals::mus(i));
    }

    r_float_.resize(globals::num_spins, 3);
    for (auto i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
          r_float_(i, j) = globals::lattice->lattice_site_position_cart(i)[j];
        }
    }
}

// --------------------------------------------------------------------------

jams::Real CudaDipoleBruteforceHamiltonian::calculate_total_energy(jams::Real time) {
    double e_total = 0.0;

    calculate_fields(time);
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += -0.5 * (  globals::s(i,0)*field_(i,0)
                           + globals::s(i,1)*field_(i,1)
                           + globals::s(i,2)*field_(i,2) );
    }

    return e_total;
}

jams::Real CudaDipoleBruteforceHamiltonian::calculate_one_spin_energy(const int i, const Vec3 &s_i, jams::Real time) {
    const auto field = calculate_field(i, time);
    return -jams::dot(s_i, field) / 2;
}

jams::Real CudaDipoleBruteforceHamiltonian::calculate_energy(const int i, jams::Real time) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i, time);
}

jams::Real CudaDipoleBruteforceHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) {
    const auto field = calculate_field(i, time);
    double e_initial = -jams::dot(spin_initial, field);
    double e_final = -jams::dot(spin_final, field);
    return 0.5*(e_final - e_initial);
}

void CudaDipoleBruteforceHamiltonian::calculate_energies(jams::Real time) {
    for (auto i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_energy(i, time);
    }
}

Vec3R CudaDipoleBruteforceHamiltonian::calculate_field(const int i, jams::Real time) {
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
          globals::lattice->lattice_site_position_cart(i),
          globals::lattice->lattice_site_position_cart(j));
  };

  const auto r_cut_squared = r_cutoff_ * r_cutoff_;
  const jams::Real w0 = globals::mus(i) * static_cast<jams::Real>(kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter())));

  jams::Real hx = 0, hy = 0, hz = 0;
#if HAS_OMP
#pragma omp parallel for reduction(+:hx, hy, hz)
#endif
  for (auto j = 0; j < globals::num_spins; ++j) {
    if (j == i) continue;

    const Vec3 s_j = {globals::s(j,0), globals::s(j,1), globals::s(j,2)};

    Vec3 r_ij = displacement(i, j);

    const jams::Real r_abs_sq = jams::norm_squared(r_ij);

      const auto eps = static_cast<jams::Real>(jams::defaults::lattice_tolerance*jams::defaults::lattice_tolerance);
    if (definately_greater_than(r_abs_sq, r_cut_squared, eps)) continue;

    hx += w0 * globals::mus(j) * (3.0 * r_ij[0] * jams::dot(s_j, r_ij) -
        jams::norm_squared(r_ij) * s_j[0]) / pow5(jams::norm(r_ij));
    hy += w0 * globals::mus(j) * (3.0 * r_ij[1] * jams::dot(s_j, r_ij) -
        jams::norm_squared(r_ij) * s_j[1]) / pow5(jams::norm(r_ij));;
    hz += w0 * globals::mus(j) * (3.0 * r_ij[2] * jams::dot(s_j, r_ij) -
        jams::norm_squared(r_ij) * s_j[2]) / pow5(jams::norm(r_ij));;
  }

  return {hx, hy, hz};
}

void CudaDipoleBruteforceHamiltonian::calculate_fields(jams::Real time) {
    CudaStream stream;

    DipoleBruteforceKernel<<<(globals::num_spins + block_size - 1)/block_size, block_size, 0, stream.get() >>>
        (globals::s.device_data(), r_float_.device_data(), mus_float_.device_data(), globals::num_spins, field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
