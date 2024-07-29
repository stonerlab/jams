#include <jams/hamiltonian/cuda_crystal_field.h>
#include <jams/hamiltonian/cuda_crystal_field_kernel.cuh>

#include <jams/core/globals.h>
#include <jams/cuda/cuda_array_reduction.h>

CudaCrystalFieldHamiltonian::CudaCrystalFieldHamiltonian(
        const libconfig::Setting &settings, const unsigned int size) : CrystalFieldHamiltonian(
        settings, size) {}

void CudaCrystalFieldHamiltonian::calculate_fields(double time) {
    dim3 block_size;
    block_size.x = 64;

    dim3 grid_size;
    grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

    cuda_crystal_field_kernel<<<grid_size, block_size, 0, dev_stream_.get() >>>
            (globals::num_spins, globals::s.device_data(), crystal_field_tesseral_coeff_.device_data(), field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaCrystalFieldHamiltonian::calculate_energies(double time) {
    dim3 block_size;
    block_size.x = 64;

    dim3 grid_size;
    grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

    cuda_crystal_field_energy_kernel<<<grid_size, block_size, 0, dev_stream_.get() >>>
            (globals::num_spins, globals::s.device_data(), crystal_field_tesseral_coeff_.device_data(), energy_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

double CudaCrystalFieldHamiltonian::calculate_total_energy(double time) {
    calculate_energies(time);
    return jams::scalar_field_reduce_cuda(energy_);
}
