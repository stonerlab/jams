#ifndef JAMS_CORE_EXCEPTION_H
#define JAMS_CORE_EXCEPTION_H

#include <stdexcept>
#include <sstream>
#include <cstring>

#if HAS_CUDA
#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>
#endif

namespace jams {

  class runtime_error : std::runtime_error {
      std::string msg;
  public:
      runtime_error(const std::string &arg, const char *file, int line, const char *function="") :
      std::runtime_error(arg) {
          std::ostringstream o;
          if (strcmp(function, "") == 0) {
              o << file << ":" << line << ": " << arg ;
          } else {
              o << file << ":" << line << "[" << function << "]" << ": " << arg ;
          }
          msg = o.str();
      }
      ~runtime_error() throw() {}
      const char *what() const throw() {
          return msg.c_str();
      }
  };

  class unimplemented_error : public std::logic_error
  {
  public:
      explicit unimplemented_error(const std::string &func) : std::logic_error("unimplemented function: " + func) { };
  };
#define JAMS_UNIMPLEMENTED_FUNCTION throw jams::unimplemented_error(__PRETTY_FUNCTION__)
}

#if HAS_CUDA
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
#endif

#endif  // JAMS_CORE_EXCEPTION_H
