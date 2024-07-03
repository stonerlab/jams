//
// Created by Joseph Barker on 2020-03-09.
//

#ifndef JAMS_COMMON_H
#define JAMS_COMMON_H

#include <sstream>
#include <pcg_random.hpp>
#include <jams/interface/randutils.h>
#include "jams/interface/system.h"

#if HAS_CUDA
#include <cusparse.h>
#include <curand.h>
#include <cublas_v2.h>

#endif

namespace jams {
    class Jams;
    Jams& instance();

    using RandomGeneratorType = pcg32;

    enum class Mode {CPU, GPU};

    class Jams {
    public:

        Jams();
        ~Jams();

        // disable copy and assign
        Jams(const Jams&) = delete;
        void operator=(const Jams&) = delete;


        inline static Mode mode() { return instance().mode_; }
        static void set_mode(Mode mode);

        inline static const std::string& output_path() { return instance().output_path_; }
        static void set_output_dir(const std::string& path) {
          instance().output_path_ = path;
          jams::system::make_path(path);
        }

        inline static RandomGeneratorType& random_generator() { return instance().random_generator_; }

        inline static std::string random_generator_internal_state() {
          std::stringstream ss;
          ss << instance().random_generator_;
          return ss.str();
        }

        static bool has_gpu_device();

        #ifdef HAS_CUDA
        inline static cublasHandle_t cublas_handle() { return instance().cublas_handle_; }
        inline static cusparseHandle_t cusparse_handle() { return instance().cusparse_handle_; }
        inline static curandGenerator_t curand_generator() { return instance().curand_generator_; }
        #endif

    private:
        void init_device_handles();
        void make_output_dir();

        Mode mode_ = Mode::CPU;

        RandomGeneratorType random_generator_{randutils::auto_seed_128{}.base()};

        std::string output_path_ = ".";

        #if HAS_CUDA
        cublasHandle_t cublas_handle_ = nullptr;
        cusparseHandle_t cusparse_handle_ = nullptr;
        curandGenerator_t curand_generator_ = nullptr;
        #endif
    };
}

#endif //JAMS_COMMON_H
