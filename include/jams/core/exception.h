#ifndef JAMS_CORE_EXCEPTION_H
#define JAMS_CORE_EXCEPTION_H

#include <stdexcept>
#include <sstream>
#include <cstring>

#ifdef CUDA
#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>
#endif

class general_exception : std::runtime_error {
    std::string msg;
public:
    general_exception(const std::string &arg, const char *file, int line, const char *function="") :
    std::runtime_error(arg) {
        std::ostringstream o;
        if (strcmp(function, "") == 0) {
            o << file << ":" << line << ": " << arg ;
        } else {
            o << file << ":" << line << "[" << function << "]" << ": " << arg ;
        }
        msg = o.str();
    }
    ~general_exception() throw() {}
    const char *what() const throw() {
        return msg.c_str();
    }
};

class unimplemented_exception : general_exception {
    public:
        unimplemented_exception(const char *file, int line, const char *function="")
        : general_exception("unimplemented", file, line, function) {}
};

class untested_exception : general_exception {
    public:
        untested_exception(const char *file, int line, const char *function="")
        : general_exception("untested", file, line, function) {}
};

#ifdef CUDA
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
