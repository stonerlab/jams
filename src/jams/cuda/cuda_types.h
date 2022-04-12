// cuda_types.h                                                        -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_TYPES
#define INCLUDED_JAMS_CUDA_TYPES

#include <complex>

#include <cuda.h>
#include <cusparse.h>

namespace jams {
namespace cuda {

// Functions to map templated types onto the correct CUDA types.
template<typename T>
inline constexpr cudaDataType get_cuda_data_type();

template<>
inline constexpr cudaDataType get_cuda_data_type<uint8_t>() {
  return CUDA_R_8U;
}

template<>
inline constexpr cudaDataType get_cuda_data_type<int8_t>() {
  return CUDA_R_8I;
}

template<>
inline constexpr cudaDataType get_cuda_data_type<int32_t>() {
  return CUDA_R_32I;
}

template<>
inline constexpr cudaDataType get_cuda_data_type<float>() {
  return CUDA_R_32F;
}

template<>
inline constexpr cudaDataType get_cuda_data_type<double>() {
  return CUDA_R_64F;
}

template<>
inline constexpr cudaDataType get_cuda_data_type<std::complex<float>>() {
  return CUDA_C_32F;
}

template<>
inline constexpr cudaDataType get_cuda_data_type<std::complex<double>>() {
  return CUDA_C_64F;
}

// Functions to map templated types onto the correct cuSPARSE types.
template<typename T>
inline constexpr cusparseIndexType_t get_cusparse_index_type();

template<>
inline constexpr cusparseIndexType_t get_cusparse_index_type<uint16_t>() {
  return CUSPARSE_INDEX_16U;
}

template<>
inline constexpr cusparseIndexType_t get_cusparse_index_type<int32_t>() {
  return CUSPARSE_INDEX_32I;
}

template<>
inline constexpr cusparseIndexType_t get_cusparse_index_type<int64_t>() {
  return CUSPARSE_INDEX_64I;
}
}
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------