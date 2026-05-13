#include <jams/hamiltonian/cuda_crystal_field.h>
#include <jams/hamiltonian/cuda_crystal_field_kernel.cuh>

#include <jams/core/globals.h>
#include <jams/cuda/cuda_array_reduction.h>
#include <jams/cuda/cuda_common.h>

CudaCrystalFieldHamiltonian::CudaCrystalFieldHamiltonian(
        const libconfig::Setting &settings, const unsigned int size) : CrystalFieldHamiltonian(
        settings, size) {}

void CudaCrystalFieldHamiltonian::calculate_fields(jams::Real time) {
    dim3 block_size;
    block_size.x = 64;

    CHECK_CUDA_STATUS(cudaMemsetAsync(
        field_.mutable_device_data(),
        0,
        field_.bytes(),
        cuda_stream_.get()));

    const auto active_spin_count = active_spin_indices_.elements();
    if (active_spin_count == 0) {
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
        return;
    }

    dim3 grid_size;
    grid_size.x = (active_spin_count + block_size.x - 1) / block_size.x;

    cuda_crystal_field_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
            (active_spin_count,
             active_spin_indices_.device_data(),
             globals::s.device_data(),
             spin_profile_.device_data(),
             u_axes_.device_data(),
             v_axes_.device_data(),
             w_axes_.device_data(),
             profile_axis_modes_.device_data(),
             profile_pointer_.device_data(),
             tesseral_keys_.device_data(),
             tesseral_coefficients_.device_data(),
             axial_polynomial_coefficients_.device_data(),
             field_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaCrystalFieldHamiltonian::calculate_energies(jams::Real time) {
    dim3 block_size;
    block_size.x = 64;

    CHECK_CUDA_STATUS(cudaMemsetAsync(
        energy_.mutable_device_data(),
        0,
        energy_.bytes(),
        cuda_stream_.get()));

    const auto active_spin_count = active_spin_indices_.elements();
    if (active_spin_count == 0) {
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
        return;
    }

    dim3 grid_size;
    grid_size.x = (active_spin_count + block_size.x - 1) / block_size.x;

    cuda_crystal_field_energy_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
            (active_spin_count,
             active_spin_indices_.device_data(),
             globals::s.device_data(),
             spin_profile_.device_data(),
             u_axes_.device_data(),
             v_axes_.device_data(),
             w_axes_.device_data(),
             profile_axis_modes_.device_data(),
             profile_pointer_.device_data(),
             tesseral_keys_.device_data(),
             tesseral_coefficients_.device_data(),
             axial_polynomial_coefficients_.device_data(),
             energy_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

jams::Real CudaCrystalFieldHamiltonian::calculate_total_energy(jams::Real time) {
    calculate_energies(time);
    return jams::scalar_field_reduce_cuda(energy_, cuda_stream_.get());
}
