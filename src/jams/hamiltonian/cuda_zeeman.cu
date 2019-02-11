#include "jams/hamiltonian/cuda_zeeman.h"
#include "jams/hamiltonian/cuda_zeeman_kernel.cuh"

#include "jblib/containers/cuda_array.h"
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

  dev_energy_ = jblib::CudaArray<double, 1>(energy_);
  dev_field_  = jblib::CudaArray<double, 1>(field_);

  dev_dc_local_field_ = jblib::CudaArray<double, 1>(dc_local_field_);

  dev_ac_local_field_ = jblib::CudaArray<double, 1>(ac_local_field_);
  dev_ac_local_frequency_ = jblib::CudaArray<double, 1>(ac_local_frequency_);
}

void CudaZeemanHamiltonian::calculate_fields() {
    dim3 block_size;
        block_size.x = 32;
        block_size.y = 4;

        dim3 grid_size;
        grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;
        grid_size.y = (3 + block_size.y - 1) / block_size.y;

        cudaMemcpyAsync(dev_field_.data(),           // void *               dst
                   dev_dc_local_field_.data(),               // const void *         src
                   globals::num_spins3*sizeof(double),   // size_t               count
                   cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                   dev_stream_);                   // device stream
        DEBUG_CHECK_CUDA_ASYNC_STATUS;

        if (has_ac_local_field_) {
            cuda_zeeman_ac_field_kernel<<<grid_size, block_size, 0, dev_stream_>>>
                (globals::num_spins, solver->time(),
                    dev_ac_local_field_.data(), dev_ac_local_frequency_.data(),
                    solver->dev_ptr_spin(), dev_field_.data());
            DEBUG_CHECK_CUDA_ASYNC_STATUS;
        }
}

CudaZeemanHamiltonian::~CudaZeemanHamiltonian() {
  if (dev_stream_ != nullptr) {
    CHECK_CUDA_STATUS(cudaStreamDestroy(dev_stream_));
  }
}