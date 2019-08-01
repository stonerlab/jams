#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "cuda_dipole_bruteforce.h"
#include "cuda_dipole_bruteforce_kernel.cuh"

CudaDipoleHamiltonianBruteforce::~CudaDipoleHamiltonianBruteforce() {
}

CudaDipoleHamiltonianBruteforce::CudaDipoleHamiltonianBruteforce(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size) {
    Vec3 super_cell_dim = {0.0, 0.0, 0.0};

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice->size(n));
    }

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

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
    for (int i = 0; i < globals::num_spins; ++i) {
      mus_float_(i) = static_cast<float>(globals::mus(i));
    }

    r_float_.resize(globals::num_spins, 3);
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
          r_float_(i, j) = lattice->atom_position(i)[j];
        }
    }

    dipole_fields_.resize(globals::num_spins, 3);
}

// --------------------------------------------------------------------------

double CudaDipoleHamiltonianBruteforce::calculate_total_energy() {
    double e_total = 0.0;

    calculate_fields(dipole_fields_);
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += -0.5 * (  globals::s(i,0)*dipole_fields_(i,0)
                           + globals::s(i,1)*dipole_fields_(i,1)
                           + globals::s(i,2)*dipole_fields_(i,2) );
    }

    return e_total;
}

double CudaDipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -0.5 * (s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

double CudaDipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i);
}

double CudaDipoleHamiltonianBruteforce::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}

void CudaDipoleHamiltonianBruteforce::calculate_energies(jams::MultiArray<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies(i) = calculate_one_spin_energy(i);
    }
}

void CudaDipoleHamiltonianBruteforce::calculate_one_spin_field(const int i, double h[3]) {

    h[0] = 0.0; h[1] = 0.0; h[2] = 0.0;

    for (auto j = 0; j < globals::num_spins; ++j) {
        if (j == i) continue;

        auto r_ij = lattice->displacement(i, j);
        const auto r_abs_sq = norm_sq(r_ij);

        if (r_abs_sq > (r_cutoff_*r_cutoff_)) continue;

        const auto r_abs = sqrt(r_abs_sq);
        const auto w0 = dipole_prefactor_ * globals::mus(i) * globals::mus(j) / pow5(r_abs);
        const Vec3 s_j = {globals::s(j, 0), globals::s(j, 1), globals::s(j, 2)};
        const auto s_j_dot_rhat = 3.0 * dot(s_j, r_ij);

        #pragma unroll
        for (auto n = 0; n < 3; ++n) {
            h[n] += w0 * (r_ij[n] * s_j_dot_rhat - r_abs_sq * s_j[n]);
        }
    }
}

void CudaDipoleHamiltonianBruteforce::calculate_fields(jams::MultiArray<double, 2>& fields) {
    CudaStream stream;

    DipoleBruteforceKernel<<<(globals::num_spins + block_size - 1)/block_size, block_size, 0, stream.get() >>>
        (globals::s.device_data(), r_float_.device_data(), mus_float_.device_data(), globals::num_spins, fields.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
