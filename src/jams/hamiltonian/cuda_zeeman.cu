#include "jams/hamiltonian/cuda_zeeman.h"

#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"
#include "jams/cuda/cuda_common.h"

__global__ void cuda_zeeman_energy_kernel(const unsigned int num_spins, const jams::Real time, const jams::Real * dc_local_field,
  const jams::Real * ac_local_field, const jams::Real * ac_local_frequency, const jams::RealHi * dev_s, jams::Real * dev_e) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int base = 3 * idx;
    if (idx >= num_spins) return;

    const jams::Real s[3] = {static_cast<jams::Real>(dev_s[base + 0]), static_cast<jams::Real>(dev_s[base + 1]), static_cast<jams::Real>(dev_s[base + 2])};

    jams::Real e_total = 0.0;
    for (unsigned int n = 0; n < 3; ++n) {
        e_total += s[n]  * (dc_local_field[3 * idx + n] + ac_local_field[3 * idx + n] * cos(ac_local_frequency[idx] * time));
    }
    dev_e[idx] = e_total;
}

__global__ void cuda_zeeman_ac_field_kernel(const unsigned int num_spins, const jams::Real time,
  const jams::Real * ac_local_field, const jams::Real * ac_local_frequency, jams::Real * dev_h) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int gxy = 3 * idx + idy;

    if (idx < num_spins && idy < 3) {
        dev_h[gxy] += ac_local_field[gxy] * cos(ac_local_frequency[idx] * time);
    }
}

CudaZeemanHamiltonian::CudaZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : ZeemanHamiltonian(settings, size) {
}

void CudaZeemanHamiltonian::calculate_fields(jams::Real time) {
    dim3 block_size;
        block_size.x = 32;
        block_size.y = 4;

        dim3 grid_size;
        grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;
        grid_size.y = (3 + block_size.y - 1) / block_size.y;

        cudaMemcpyAsync(field_.device_data(),           // void *               dst
                   dc_local_field_.device_data(),               // const void *         src
                   dc_local_field_.bytes(),   // size_t               count
                   cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                   cuda_stream_.get());                   // device stream
        DEBUG_CHECK_CUDA_ASYNC_STATUS;

        if (has_ac_local_field_) {
            cuda_zeeman_ac_field_kernel<<<grid_size, block_size, 0, cuda_stream_.get()>>>
                (globals::num_spins, time,
                    ac_local_field_.device_data(), ac_local_frequency_.device_data(), field_.device_data());
            DEBUG_CHECK_CUDA_ASYNC_STATUS;
        }
}
