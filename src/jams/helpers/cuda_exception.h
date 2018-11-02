//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_EXCEPTION_H
#define JAMS_CUDA_EXCEPTION_H

#include <stdexcept>
#include <sstream>
#include <cstring>

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

class cuda_api_exception : std::runtime_error {
    std::string msg;
public:
    cuda_api_exception(const std::string &arg, const char *file, int line, const char *function="") :
            std::runtime_error(arg) {
      cudaError_t error = cudaPeekAtLastError();
      std::ostringstream o;
      if (strcmp(function, "") == 0) {
        o << file << ":" << line << ": " << arg << "\n";
      } else {
        o << file << ":" << line << "[" << function << "]" << ": " << arg <<"\n";
      }
      o << "::" << cudaGetErrorString(error) << "(" << error << ")" << std::endl;
      msg = o.str();
    }
    ~cuda_api_exception() throw() {}
    const char *what() const throw() {
      return msg.c_str();
    }
};

#endif //JAMS_CUDA_EXCEPTION_H
