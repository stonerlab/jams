#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/core/output.h"

#include "jams/hamiltonian/dipole_bruteforce.h"
#include "jams/hamiltonian/dipole_bruteforce_kernel.h"

DipoleHamiltonianBruteforce::DipoleHamiltonianBruteforce(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size) {
    jblib::Vec3<double> super_cell_dim(0.0, 0.0, 0.0);

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice->size(n));
    }

    r_cutoff_ = *std::max_element(super_cell_dim.begin(), super_cell_dim.end());

    settings.lookupValue("r_cutoff", r_cutoff_);
    output->write("  r_cutoff: %e\n", r_cutoff_);

    dipole_prefactor_ = kVacuumPermeadbility*kBohrMagneton /(4*kPi*::lattice->parameter() * ::lattice->parameter() * ::lattice->parameter());


#ifdef CUDA
    if (solver->is_cuda_solver()) {
    bool super_cell_pbc[3];
    float super_unit_cell[3][3];
    float super_unit_cell_inv[3][3];

    for (int i = 0; i < 3; ++i) {
        super_cell_pbc[i] = ::lattice->is_periodic(i);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            super_unit_cell[i][j] = ::lattice->unit_cell_vector(i)[j] * ::lattice->size(i);
        }
    }
    matrix_invert(super_unit_cell, super_unit_cell_inv);

    float r_cutoff_float = r_cutoff_;

    float f_dipole_prefactor = dipole_prefactor_;

    cudaMemcpyToSymbol(dev_dipole_prefactor,    &f_dipole_prefactor,       sizeof(float));
    cudaMemcpyToSymbol(dev_r_cutoff,           &r_cutoff_float,       sizeof(float));
    cudaMemcpyToSymbol(dev_super_cell_pbc,      super_cell_pbc,      3 * sizeof(bool));
    cudaMemcpyToSymbol(dev_super_unit_cell,     super_unit_cell,     9 * sizeof(float));
    cudaMemcpyToSymbol(dev_super_unit_cell_inv, super_unit_cell_inv, 9 * sizeof(float));

    jblib::Array<float, 1> f_mus(globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
      f_mus[i] = globals::mus[i];
    }

    dev_mus_ = jblib::CudaArray<float, 1>(f_mus);

    jblib::Array<float, 2> r(globals::num_spins, 3);

    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
            r(i, j) = lattice->atom_position(i)[j];
        }
    }

    dev_r_ = jblib::CudaArray<float, 1>(r);

    cudaStreamCreate(&dev_stream_);
    }
#endif  // CUDA

}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_total_energy() {
    double e_total = 0.0;

#ifdef CUDA
    static jblib::CudaArray<double, 1> dev_dipole_fields(globals::num_spins*3);
    static jblib::Array<double, 2> host_dipole_fields(globals::num_spins, 3);

   if (solver->is_cuda_solver()) {
        calculate_fields(dev_dipole_fields);
        dev_dipole_fields.copy_to_host_array(host_dipole_fields);
        for (int i = 0; i < globals::num_spins; ++i) {
            e_total += -0.5 * (  globals::s(i,0)*host_dipole_fields(i,0) 
                               + globals::s(i,1)*host_dipole_fields(i,1)
                               + globals::s(i,2)*host_dipole_fields(i,2) );
        }
   } else { 
#endif // CUDA
       for (int i = 0; i < globals::num_spins; ++i) {
           e_total += calculate_one_spin_energy(i);
       }
#ifdef CUDA
    }
#endif // CUDA

    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -0.5 * (s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_one_spin_field(const int i, double h[3]) {
    int n,j;
    jblib::Vec3<double> field;

    const double prefactor = globals::mus(i) * dipole_prefactor_;

    h[0] = 0.0; h[1] = 0.0; h[2] = 0.0;

    for (j = 0; j < globals::num_spins; ++j) {
        if (j == i) continue;

        auto r_ij = lattice->displacement(i, j);

        const auto r_abs_sq = r_ij.norm_sq();

        if (r_abs_sq > (r_cutoff_*r_cutoff_)) continue;

        const auto r_abs = sqrt(r_abs_sq);

        const auto w0 = prefactor * globals::mus(j) / (r_abs_sq * r_abs_sq * r_abs);

        const jblib::Vec3<double> s_j = {globals::s(j, 0), globals::s(j, 1), globals::s(j, 2)};
        
        const auto s_j_dot_rhat = 3.0 * dot(s_j, r_ij);

        #pragma unroll
        for (n = 0; n < 3; ++n) {
            h[n] += w0 * (r_ij[n] * s_j_dot_rhat - r_abs_sq * s_j[n]);
        }
    }

    // for (j = 0; j < globals::num_spins; ++j) {
    //     if (j == i) continue;

    //     auto r_ij = lattice->displacement(i, j);

    //     auto r_abs = r_ij.norm();

    //     if (r_abs > r_cutoff_) continue;

    //     auto w0 = prefactor * globals::mus(j);

    //     const jblib::Vec3<double> s_j = {globals::s(j, 0), globals::s(j, 1), globals::s(j, 2)};
    //     auto s_j_dot_rhat = dot(s_j, r_ij);

    //     #pragma unroll
    //     for (n = 0; n < 3; ++n) {
    //         h[n] += (3.0 * r_ij[n] * s_j_dot_rhat - r_abs * r_abs * s_j[n]) * w0 / (r_abs*r_abs*r_abs*r_abs*r_abs);
    //     }
    // }


}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_fields(jblib::Array<double, 2>& fields) {
    for (int i = 0; i < globals::num_spins; ++i) {
        double h[3];

        calculate_one_spin_field(i, h);

        for (int n = 0; n < 3; ++n) {
            fields(i, n) = h[n];
        }
    }
}

void DipoleHamiltonianBruteforce::calculate_fields(jblib::CudaArray<double, 1>& fields) {
    DipoleBruteforceKernel<<<(globals::num_spins + block_size - 1)/block_size, block_size, 0, dev_stream_ >>>
        (solver->dev_ptr_spin(), dev_r_.data(), dev_mus_.data(), globals::num_spins, fields.data()); 

	// dipole_bruteforce_sharemem_kernel<<<(globals::num_spins + block_size - 1)/block_size, block_size, 0, dev_stream_ >>>
	    // (solver->dev_ptr_spin(), dev_r_.data(), dev_mus_.data(), globals::num_spins, fields.data()); 

    // dipole_bruteforce_kernel<<<(globals::num_spins+block_size-1)/block_size, block_size, 0, dev_stream_ >>>
        // (solver->dev_ptr_spin(), dev_r_.data(), dev_mus_.data(), globals::num_spins, fields.data()); 
}

// --------------------------------------------------------------------------
