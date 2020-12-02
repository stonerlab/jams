//
// Created by Joseph Barker on 2020-03-09.
//

#include <cassert>
#include <iostream>

#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/random.h"

#include "jams/common.h"

namespace jams {
    Jams &instance() {
      static Jams jams_instance;
      return jams_instance;
    }


    #ifdef HAS_CUDA

    Jams::Jams() :
      random_generator_({randutils::auto_seed_128{}.base()})
    {
    }

    Jams::~Jams() {
      if (cublas_handle_ != nullptr) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
      }

      if (cusparse_handle_ != nullptr) {
        cusparseDestroy(cusparse_handle_);
        cusparse_handle_ = nullptr;
      }

      if (curand_generator_ != nullptr) {
        curandDestroyGenerator(curand_generator_);
        curand_generator_ = nullptr;
      }

    }

    bool Jams::has_gpu_device() {
      bool r = (cudaSuccess == cudaFree(0));
      cudaGetLastError();
      return r;
    }

    void Jams::set_mode(Mode mode) {
      jams::instance().mode_ = mode;

      if (mode == Mode::GPU) {
        jams::instance().init_device_handles();
      }
    }

    void Jams::init_device_handles() {
      if (cusparse_handle_ == nullptr) {
        if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
          throw std::runtime_error("Failed to initialise cusparse");
        }
      }

      if (cublas_handle_ == nullptr) {
        CHECK_CUBLAS_STATUS(cublasCreate(&cublas_handle_));
      }

      if (curand_generator_ == nullptr) {
        uint64_t dev_rng_seed = jams::instance().random_generator()();

        CHECK_CURAND_STATUS(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(curand_generator_, dev_rng_seed));
        CHECK_CURAND_STATUS(curandGenerateSeeds(curand_generator_));
      }
    }


    #else
    Jams::Jams() {

    }

    Jams::~Jams() {
    }


    void Jams::set_mode(Mode mode) {
      jams::instance().mode_ = mode;

      if (mode == Mode::GPU) {
        throw std::runtime_error("GPU mode not possible in CPU build");
      }
    }

    bool Jams::has_gpu_device() {
      return false;
    }
    #endif
}