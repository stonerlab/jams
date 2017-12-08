#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/error.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"

#include "strategy.h"

#include "cuda_dipole_fft.h"
#include "dipole.h"
#include "dipole_bruteforce.h"
#include "dipole_cpu_bruteforce.h"
#include "dipole_tensor.h"
#include "dipole_cuda_sparse_tensor.h"
#include "dipole_ewald.h"
#include "dipole_fft.h"

#include "jblib/containers/array.h"

DipoleHamiltonian::DipoleHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {

#ifdef CUDA
    if (solver->is_cuda_solver()) {
        dev_energy_.resize(globals::num_spins);
        dev_field_.resize(globals::num_spins3);
    }
#endif

    dipole_strategy_ = select_strategy(settings, size);
}

// --------------------------------------------------------------------------

HamiltonianStrategy * DipoleHamiltonian::select_strategy(const libconfig::Setting &settings, const unsigned int size) {
    if (settings.exists("strategy")) {
        std::string strategy_name(capitalize(settings["strategy"]));

        if (strategy_name == "TENSOR") {
            return new DipoleHamiltonianTensor(settings, size);
        }

        if (strategy_name == "CUDA_SPARSE_TENSOR") {
            return new DipoleHamiltonianCUDASparseTensor(settings, size);
        }

        if (strategy_name == "EWALD") {
            return new DipoleHamiltonianEwald(settings, size);
        }

        if (strategy_name == "FFT") {
            if (solver->is_cuda_solver()) {
                return new CudaDipoleHamiltonianFFT(settings, size);
            }
            return new DipoleHamiltonianFFT(settings, size);
        }

        if (strategy_name == "BRUTEFORCE") {
          if (solver->is_cuda_solver()) {
            return new DipoleHamiltonianBruteforce(settings, size);
          }
          return new DipoleHamiltonianCpuBruteforce(settings, size);
        }

        std::runtime_error("Unknown DipoleHamiltonian strategy '" + strategy_name + "' requested\n");
    }
    jams_warning("no dipole strategy selected, defaulting to TENSOR");
    return new DipoleHamiltonianTensor(settings, size);
}

// --------------------------------------------------------------------------

double DipoleHamiltonian::calculate_total_energy() {
    return dipole_strategy_->calculate_total_energy();
}

// --------------------------------------------------------------------------

double DipoleHamiltonian::calculate_one_spin_energy(const int i) {
    return dipole_strategy_->calculate_one_spin_energy(i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    return dipole_strategy_->calculate_one_spin_energy_difference(i, spin_initial, spin_final);
}
// --------------------------------------------------------------------------

void DipoleHamiltonian::calculate_energies() {
    dipole_strategy_->calculate_energies(energy_);
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::calculate_one_spin_field(const int i, double h[3]) {
    dipole_strategy_->calculate_one_spin_field(i, h);
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::calculate_fields() {
    if (solver->is_cuda_solver()) {
        dipole_strategy_->calculate_fields(dev_field_);
    } else {
        dipole_strategy_->calculate_fields(field_);
    }
}

