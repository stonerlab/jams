#include "jams/hamiltonian/cuda_zeeman.h"
#include "jams/hamiltonian/cuda_zeeman_kernel.cuh"

#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"
#include "jams/cuda/cuda_common.h"

CudaZeemanHamiltonian::CudaZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : ZeemanHamiltonian(settings, size) {
  CHECK_CUDA_STATUS(cudaStreamCreate(&dev_stream_));
}

void CudaZeemanHamiltonian::calculate_fields() {
    dim3 block_size;
        block_size.x = 32;
        block_size.y = 4;

        dim3 grid_size;
        grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;
        grid_size.y = (3 + block_size.y - 1) / block_size.y;

        cudaMemcpyAsync(field_.device_data(),           // void *               dst
                   dc_local_field_.device_data(),               // const void *         src
                   globals::num_spins3*sizeof(double),   // size_t               count
                   cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                   dev_stream_);                   // device stream
        DEBUG_CHECK_CUDA_ASYNC_STATUS;

        if (has_ac_local_field_) {
            cuda_zeeman_ac_field_kernel<<<grid_size, block_size, 0, dev_stream_>>>
                (globals::num_spins, solver->time(),
                    ac_local_field_.device_data(), ac_local_frequency_.device_data(),
                    globals::s.device_data(), field_.device_data());
            DEBUG_CHECK_CUDA_ASYNC_STATUS;
        }
}

CudaZeemanHamiltonian::~CudaZeemanHamiltonian() {
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}