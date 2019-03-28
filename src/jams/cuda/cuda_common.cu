#include "jams/cuda/cuda_common.h"

#include <cusparse.h>
#include <curand.h>

const char* cusparseGetStatusString(cusparseStatus_t status) {
  switch(status) {
    case CUSPARSE_STATUS_SUCCESS:                   return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:             return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }
  return "CUSPARSE_STATUS_UNKNOWN_ERROR";
}

const char* curandGetStatusString(curandStatus_t status) {
  switch(status) {
    case CURAND_STATUS_SUCCESS:                    return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:           return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:            return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:          return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:                 return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:               return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:  return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:             return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:        return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:              return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:             return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "CURAND_STATUS_UNKNOWN_ERROR";
}

const char* cublasGetStatusString(cublasStatus_t status) {
  switch(status) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

const char* cufftGetStatusString(cufftResult_t status) {
  switch(status) {
    case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:            return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:           return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:             return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:             return "CUFFT_NOT_SUPPORTED";
  }
  return "CUFFT_STATUS_UNKNOWN_ERROR";
}