//
// Created by Joseph Barker on 2019-02-06.
//

#ifndef JAMS_CUDA_SPARSE_INTERACTION_MATRIX_H
#define JAMS_CUDA_SPARSE_INTERACTION_MATRIX_H

#include <iostream>

#include <cusparse.h>

#include "jams/cuda/cuda_common.h"
#include "jams/containers/sparsematrix.h"

template<typename T>
struct CudaSparseMatrixCSR {
    int num_rows = 0;
    int num_cols = 0;
    int num_nonzero = 0;
    int *row = nullptr;
    int *col = nullptr;
    T   *val = nullptr;
    cusparseMatDescr_t descr = nullptr;
};

// This function supports mixed precision by converting the host matrix type
template<typename THst, typename TDev>
void sparsematrix_copy_host_csr_to_cuda_csr(const SparseMatrix<THst>& host_matrix, CudaSparseMatrixCSR<TDev>& cuda_matrix) {
  assert(host_matrix.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR);

  if (!cuda_matrix.descr) {
    CHECK_CUSPARSE_STATUS(cusparseCreateMatDescr(&cuda_matrix.descr));
  }

  switch(host_matrix.getMatrixType()) {
    case SPARSE_MATRIX_TYPE_GENERAL:
      cusparseSetMatType(cuda_matrix.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
      break;
    case SPARSE_MATRIX_TYPE_SYMMETRIC:
      cusparseSetMatType(cuda_matrix.descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
      break;
    case SPARSE_MATRIX_TYPE_HERMITIAN:
      cusparseSetMatType(cuda_matrix.descr, CUSPARSE_MATRIX_TYPE_HERMITIAN);
      break;
    case SPARSE_MATRIX_TYPE_TRIANGULAR:
      cusparseSetMatType(cuda_matrix.descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
      break;
    default:
      throw std::runtime_error("Unsupported host matix type");
  }

  CHECK_CUSPARSE_STATUS(cusparseSetMatIndexBase(cuda_matrix.descr, CUSPARSE_INDEX_BASE_ZERO));

  cuda_matrix.num_rows = host_matrix.rows();
  cuda_matrix.num_cols = host_matrix.cols();
  cuda_matrix.num_nonzero = host_matrix.nonZero();

  CHECK_CUDA_STATUS(cudaMalloc((void**)&cuda_matrix.row, (host_matrix.rows()+1)*sizeof(int)));
  CHECK_CUDA_STATUS(cudaMalloc((void**)&cuda_matrix.col, (host_matrix.nonZero())*sizeof(int)));
  CHECK_CUDA_STATUS(cudaMemcpy(cuda_matrix.row, host_matrix.rowPtr(),
                               (host_matrix.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA_STATUS(cudaMemcpy(cuda_matrix.col, host_matrix.colPtr(),
                               (host_matrix.nonZero())*sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA_STATUS(cudaMalloc((void**)&cuda_matrix.val, (host_matrix.nonZero())*sizeof(TDev)));

  if (sizeof(THst) == sizeof(TDev)) {
    CHECK_CUDA_STATUS(cudaMemcpy(cuda_matrix.val, host_matrix.valPtr(),
                                 (host_matrix.nonZero())*sizeof(TDev), cudaMemcpyHostToDevice));
  } else {
    // convert types
    std::vector<TDev> converted_values(host_matrix.nonZero());
    for (auto i = 0; i < host_matrix.nonZero(); ++i) {
      converted_values[i] = host_matrix.val(i);
    }

    CHECK_CUDA_STATUS(cudaMemcpy(cuda_matrix.val, converted_values.data(),
                                 (converted_values.size())*sizeof(TDev), cudaMemcpyHostToDevice));
  }
}


template <typename T>
inline constexpr cudaDataType get_cuda_data_type();

template<>
inline constexpr cudaDataType get_cuda_data_type<float>() { return CUDA_R_32F; }

template <>
inline constexpr cudaDataType get_cuda_data_type<double>() { return CUDA_R_64F; }


template <typename T>
class CudaSparseInteractionMatrix {
public:
    CudaSparseInteractionMatrix();
    ~CudaSparseInteractionMatrix();

    void create_matrix(SparseMatrix<double>& m);
    void calculate_fields(const double * spins, double * fields);
    void set_cuda_stream(cudaStream_t stream);

private:
    void allocate_buffer(const double * spins, double * fields);

    CudaSparseMatrixCSR<T> dev_csr_matrix_;
    cusparseHandle_t  cusparse_handle_ = nullptr;
    void*  dev_csr_buffer_             = nullptr;
    size_t dev_csr_buffer_size_        = 0;

#if HAS_CUSPARSE_MIXED_PREC
    // alg is a required argument even from CUDA 9, but the types are not implemented until CUDA 10
  #if __CUDACC_VER_MAJOR__ >= 10
      cusparseAlgMode_t cusparse_alg_ = CUSPARSE_ALG_NAIVE;
  #else
      cusparseAlgMode_t cusparse_alg_;
  #endif
#endif
};

template <typename T>
void CudaSparseInteractionMatrix<T>::create_matrix(SparseMatrix<double> &hst_csr_matrix) {
  sparsematrix_copy_host_csr_to_cuda_csr(hst_csr_matrix, dev_csr_matrix_);
}

template <typename T>
void CudaSparseInteractionMatrix<T>::calculate_fields(const double * dev_spins, double * dev_fields) {
  if (dev_csr_buffer_ == nullptr) {
    allocate_buffer(dev_spins, dev_fields);
  }
  assert(dev_csr_buffer_ != nullptr);

  const T one = 1.0;
  const T zero = 0.0;

  assert(cusparse_handle_ != nullptr);
  CHECK_CUSPARSE_STATUS(cusparseCsrmvEx(
          cusparse_handle_,
          cusparse_alg_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          dev_csr_matrix_.num_rows,
          dev_csr_matrix_.num_cols,
          dev_csr_matrix_.num_nonzero,
          &one, get_cuda_data_type<T>(),
          dev_csr_matrix_.descr,
          dev_csr_matrix_.val, get_cuda_data_type<T>(),
          dev_csr_matrix_.row,
          dev_csr_matrix_.col,
          dev_spins, get_cuda_data_type<double>(),
          &zero, get_cuda_data_type<T>(),
          dev_fields, get_cuda_data_type<double>(),
          get_cuda_data_type<double>(), // execution type
          dev_csr_buffer_));
}

template <typename T>
void CudaSparseInteractionMatrix<T>::allocate_buffer(const double * dev_spins, double * dev_fields) {
  assert(cusparse_handle_ != nullptr);

  T one = 1.0;
  T zero = 0.0;

  CHECK_CUSPARSE_STATUS(cusparseCsrmvEx_bufferSize(
          cusparse_handle_,
          cusparse_alg_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          dev_csr_matrix_.num_rows,
          dev_csr_matrix_.num_cols,
          dev_csr_matrix_.num_nonzero,
          &one, get_cuda_data_type<T>(),
          dev_csr_matrix_.descr,
          dev_csr_matrix_.val, get_cuda_data_type<T>(),
          dev_csr_matrix_.row,
          dev_csr_matrix_.col,
          dev_spins, get_cuda_data_type<double>(),
          &zero, get_cuda_data_type<T>(),
          dev_fields, get_cuda_data_type<double>(),
          get_cuda_data_type<double>(), // execution type
          &dev_csr_buffer_size_));

  CHECK_CUDA_STATUS(
          cudaMalloc((void**)&dev_csr_buffer_, dev_csr_buffer_size_));
}

template <typename T>
void CudaSparseInteractionMatrix<T>::set_cuda_stream(cudaStream_t stream) {
  cusparseSetStream(cusparse_handle_, stream);
}

template<typename T>
CudaSparseInteractionMatrix<T>::~CudaSparseInteractionMatrix() {
  cudaFree(dev_csr_buffer_);
}

template<typename T>
CudaSparseInteractionMatrix<T>::CudaSparseInteractionMatrix() {
  cusparseStatus_t status = cusparseCreate(&cusparse_handle_);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_die("cusparse Library initialization failed");
  }
}

#endif //JAMS_CUDA_SPARSE_INTERACTION_MATRIX_H
