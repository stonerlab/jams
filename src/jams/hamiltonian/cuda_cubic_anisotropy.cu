#include <jams/hamiltonian/cuda_cubic_anisotropy.h>
#include <jams/hamiltonian/cuda_cubic_anisotropy_kernel.cuh>

#include <jams/cuda/cuda_common.h>
#include <jams/core/globals.h>
#include <jams/hamiltonian/cubic_anisotropy.h>


CudaCubicHamiltonian::CudaCubicHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
    : CubicHamiltonian(settings, num_spins)
{}

double CudaCubicHamiltonian::calculate_total_energy(double time) {
    calculate_energies(time);
    double e_total = 0.0;
    for (auto i = 0; i < energy_.size(); ++i) {
        e_total += energy_(i);
    }
    return e_total;
}

void CudaCubicHamiltonian::calculate_energies(double time) {
    cuda_cubic_energy_kernel<<<(globals::num_spins + dev_blocksize_ - 1) / dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, order_.device_data(), magnitude_.device_data(), axis1_.device_data(),
             axis2_.device_data(), axis3_.device_data(), globals::s.device_data(), field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}


void CudaCubicHamiltonian::calculate_fields(double time) {
        cuda_cubic_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
                (globals::num_spins, order_.device_data(), magnitude_.device_data(), axis1_.device_data(),
                 axis2_.device_data(), axis3_.device_data(), globals::s.device_data(), field_.device_data());
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
