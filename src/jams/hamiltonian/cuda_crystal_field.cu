#include <jams/hamiltonian/cuda_crystal_field.h>
#include <jams/hamiltonian/cuda_crystal_field_kernel.cuh>

CudaCrystalFieldHamiltonian::CudaCrystalFieldHamiltonian(
        const libconfig::Setting &settings, const unsigned int size) : AppliedFieldHamiltonian(
        settings, size) {}

void CudaCrystalFieldHamiltonian::calculate_fields(double time) {
    dim3 block_size;
    block_size.x = 64;

    dim3 grid_size;
    grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

    cuda_crystal_field_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
            (globals::num_spins, globals::s.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}