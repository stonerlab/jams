#include <jams/hamiltonian/cuda_anisotropy_polynomial.h>

#include <jams/core/globals.h>
#include <jams/cuda/cuda_array_reduction.h>
#include <jams/cuda/cuda_common.h>
#include <jams/maths/tesseral_harmonics.h>

namespace {

__global__
void cuda_anisotropy_polynomial_energy_kernel(
    const int num_spins,
    const jams::RealHi *__restrict__ spins,
    const jams::Real *__restrict__ u_axes,
    const jams::Real *__restrict__ v_axes,
    const jams::Real *__restrict__ w_axes,
    const int *__restrict__ spin_pointer,
    const int *__restrict__ tesseral_keys,
    const jams::Real *__restrict__ tesseral_coefficients,
    jams::Real *__restrict__ energies)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spins) {
        return;
    }

    const unsigned int base = 3u * idx;
    const jams::Real sx_global = static_cast<jams::Real>(spins[base + 0]);
    const jams::Real sy_global = static_cast<jams::Real>(spins[base + 1]);
    const jams::Real sz_global = static_cast<jams::Real>(spins[base + 2]);

    const jams::Real ux = u_axes[base + 0];
    const jams::Real uy = u_axes[base + 1];
    const jams::Real uz = u_axes[base + 2];
    const jams::Real vx = v_axes[base + 0];
    const jams::Real vy = v_axes[base + 1];
    const jams::Real vz = v_axes[base + 2];
    const jams::Real wx = w_axes[base + 0];
    const jams::Real wy = w_axes[base + 1];
    const jams::Real wz = w_axes[base + 2];

    const jams::Real sx = sx_global * ux + sy_global * uy + sz_global * uz;
    const jams::Real sy = sx_global * vx + sy_global * vy + sz_global * vz;
    const jams::Real sz = sx_global * wx + sy_global * wy + sz_global * wz;

    jams::Real energy = 0;
    for (int n = spin_pointer[idx]; n < spin_pointer[idx + 1]; ++n) {
        energy += tesseral_coefficients[n]
            * jams::tesseral_monic_polynomial_key_lookup(tesseral_keys[n], sx, sy, sz);
    }

    energies[idx] = energy;
}

__global__
void cuda_anisotropy_polynomial_field_kernel(
    const int num_spins,
    const jams::RealHi *__restrict__ spins,
    const jams::Real *__restrict__ u_axes,
    const jams::Real *__restrict__ v_axes,
    const jams::Real *__restrict__ w_axes,
    const int *__restrict__ spin_pointer,
    const int *__restrict__ tesseral_keys,
    const jams::Real *__restrict__ tesseral_coefficients,
    jams::Real *__restrict__ fields)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spins) {
        return;
    }

    const unsigned int base = 3u * idx;
    const jams::Real sx_global = static_cast<jams::Real>(spins[base + 0]);
    const jams::Real sy_global = static_cast<jams::Real>(spins[base + 1]);
    const jams::Real sz_global = static_cast<jams::Real>(spins[base + 2]);

    const jams::Real ux = u_axes[base + 0];
    const jams::Real uy = u_axes[base + 1];
    const jams::Real uz = u_axes[base + 2];
    const jams::Real vx = v_axes[base + 0];
    const jams::Real vy = v_axes[base + 1];
    const jams::Real vz = v_axes[base + 2];
    const jams::Real wx = w_axes[base + 0];
    const jams::Real wy = w_axes[base + 1];
    const jams::Real wz = w_axes[base + 2];

    const jams::Real sx = sx_global * ux + sy_global * uy + sz_global * uz;
    const jams::Real sy = sx_global * vx + sy_global * vy + sz_global * vz;
    const jams::Real sz = sx_global * wx + sy_global * wy + sz_global * wz;

    jams::Real hx_local = 0;
    jams::Real hy_local = 0;
    jams::Real hz_local = 0;
    for (int n = spin_pointer[idx]; n < spin_pointer[idx + 1]; ++n) {
        jams::Real grad[3];
        jams::tesseral_monic_polynomial_grad_key_lookup(tesseral_keys[n], sx, sy, sz, grad);
        const jams::Real coeff = tesseral_coefficients[n];
        hx_local += coeff * grad[0];
        hy_local += coeff * grad[1];
        hz_local += coeff * grad[2];
    }

    const jams::Real grad_x = hx_local * ux + hy_local * vx + hz_local * wx;
    const jams::Real grad_y = hx_local * uy + hy_local * vy + hz_local * wy;
    const jams::Real grad_z = hx_local * uz + hy_local * vz + hz_local * wz;
    const jams::Real radial_grad = sx_global * grad_x + sy_global * grad_y + sz_global * grad_z;

    fields[base + 0] = radial_grad * sx_global - grad_x;
    fields[base + 1] = radial_grad * sy_global - grad_y;
    fields[base + 2] = radial_grad * sz_global - grad_z;
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
    const unsigned int num_blocks = (globals::num_spins + dev_blocksize_ - 1) / dev_blocksize_;
    cuda_anisotropy_polynomial_field_kernel<<<num_blocks, dev_blocksize_, 0, cuda_stream_.get()>>>(
        globals::num_spins,
        globals::s.device_data(),
        u_axes_.device_data(),
        v_axes_.device_data(),
        w_axes_.device_data(),
        spin_pointer_.device_data(),
        tesseral_keys_.device_data(),
        tesseral_coefficients_.device_data(),
        field_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaAnisotropyPolynomialHamiltonian::calculate_energies(jams::Real time)
{
    const unsigned int num_blocks = (globals::num_spins + dev_blocksize_ - 1) / dev_blocksize_;
    cuda_anisotropy_polynomial_energy_kernel<<<num_blocks, dev_blocksize_, 0, cuda_stream_.get()>>>(
        globals::num_spins,
        globals::s.device_data(),
        u_axes_.device_data(),
        v_axes_.device_data(),
        w_axes_.device_data(),
        spin_pointer_.device_data(),
        tesseral_keys_.device_data(),
        tesseral_coefficients_.device_data(),
        energy_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

jams::Real CudaAnisotropyPolynomialHamiltonian::calculate_total_energy(jams::Real time)
{
    calculate_energies(time);
    return jams::scalar_field_reduce_cuda(energy_, cuda_stream_.get());
}
