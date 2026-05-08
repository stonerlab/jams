#include <jams/hamiltonian/cuda_anisotropy_polynomial.h>

#include <jams/core/globals.h>
#include <jams/cuda/cuda_array_reduction.h>
#include <jams/cuda/cuda_common.h>
#include <jams/hamiltonian/tesseral_polynomial_evaluator.h>

namespace {

__global__
void cuda_anisotropy_polynomial_energy_kernel(
    const int active_spin_count,
    const int *__restrict__ active_spin_indices,
    const jams::RealHi *__restrict__ spins,
    const int *__restrict__ spin_profile,
    const jams::Real *__restrict__ u_axes,
    const jams::Real *__restrict__ v_axes,
    const jams::Real *__restrict__ w_axes,
    const int *__restrict__ profile_axis_modes,
    const int *__restrict__ profile_pointer,
    const int *__restrict__ tesseral_keys,
    const jams::Real *__restrict__ tesseral_coefficients,
    const jams::Real *__restrict__ axial_polynomial_coefficients,
    jams::Real *__restrict__ energies)
{
    const unsigned int active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_spin_count) {
        return;
    }

    const int idx = active_spin_indices[active_idx];
    const unsigned int base = 3u * idx;
    const jams::Real sx_global = static_cast<jams::Real>(spins[base + 0]);
    const jams::Real sy_global = static_cast<jams::Real>(spins[base + 1]);
    const jams::Real sz_global = static_cast<jams::Real>(spins[base + 2]);

    energies[idx] = jams::tesseral_polynomial::energy_for_spin_with_profiles(
        int(idx),
        sx_global,
        sy_global,
        sz_global,
        spin_profile,
        u_axes,
        v_axes,
        w_axes,
        profile_axis_modes,
        profile_pointer,
        tesseral_keys,
        tesseral_coefficients,
        axial_polynomial_coefficients);
}

__global__
void cuda_anisotropy_polynomial_field_kernel(
    const int active_spin_count,
    const int *__restrict__ active_spin_indices,
    const jams::RealHi *__restrict__ spins,
    const int *__restrict__ spin_profile,
    const jams::Real *__restrict__ u_axes,
    const jams::Real *__restrict__ v_axes,
    const jams::Real *__restrict__ w_axes,
    const int *__restrict__ profile_axis_modes,
    const int *__restrict__ profile_pointer,
    const int *__restrict__ tesseral_keys,
    const jams::Real *__restrict__ tesseral_coefficients,
    const jams::Real *__restrict__ axial_polynomial_coefficients,
    jams::Real *__restrict__ fields)
{
    const unsigned int active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_spin_count) {
        return;
    }

    const int idx = active_spin_indices[active_idx];
    const unsigned int base = 3u * idx;
    const jams::Real sx_global = static_cast<jams::Real>(spins[base + 0]);
    const jams::Real sy_global = static_cast<jams::Real>(spins[base + 1]);
    const jams::Real sz_global = static_cast<jams::Real>(spins[base + 2]);

    jams::Real field[3];
    jams::tesseral_polynomial::field_for_spin_with_profiles(
        int(idx),
        sx_global,
        sy_global,
        sz_global,
        spin_profile,
        u_axes,
        v_axes,
        w_axes,
        profile_axis_modes,
        profile_pointer,
        tesseral_keys,
        tesseral_coefficients,
        axial_polynomial_coefficients,
        field);

    fields[base + 0] = field[0];
    fields[base + 1] = field[1];
    fields[base + 2] = field[2];
}

} // namespace

CudaAnisotropyPolynomialHamiltonian::CudaAnisotropyPolynomialHamiltonian(
    const libconfig::Setting &settings,
    const unsigned int size)
    : AnisotropyPolynomialHamiltonian(settings, size)
{
}

void CudaAnisotropyPolynomialHamiltonian::calculate_fields(jams::Real time)
{
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        field_.mutable_device_data(),
        0,
        field_.bytes(),
        cuda_stream_.get()));

    const int active_spin_count = active_spin_indices_.elements();
    if (active_spin_count == 0) {
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
        return;
    }

    const unsigned int num_blocks = (active_spin_count + dev_blocksize_ - 1) / dev_blocksize_;
    cuda_anisotropy_polynomial_field_kernel<<<num_blocks, dev_blocksize_, 0, cuda_stream_.get()>>>(
        active_spin_count,
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

void CudaAnisotropyPolynomialHamiltonian::calculate_energies(jams::Real time)
{
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        energy_.mutable_device_data(),
        0,
        energy_.bytes(),
        cuda_stream_.get()));

    const int active_spin_count = active_spin_indices_.elements();
    if (active_spin_count == 0) {
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
        return;
    }

    const unsigned int num_blocks = (active_spin_count + dev_blocksize_ - 1) / dev_blocksize_;
    cuda_anisotropy_polynomial_energy_kernel<<<num_blocks, dev_blocksize_, 0, cuda_stream_.get()>>>(
        active_spin_count,
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

jams::Real CudaAnisotropyPolynomialHamiltonian::calculate_total_energy(jams::Real time)
{
    calculate_energies(time);
    return jams::scalar_field_reduce_cuda(energy_, cuda_stream_.get());
}
