#ifndef JAMS_CUDA_SPARSE_HELPERS_H
#define JAMS_CUDA_SPARSE_HELPERS_H

#if HAS_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "jams/containers/sparsematrix.h"
#include "cuda_defs.h"

template<typename T>
struct CudaSparseMatrixCSR {
  int     *row = nullptr;
  int     *col = nullptr;
  T       *val = nullptr;
  cusparseMatDescr_t  descr = nullptr;
};

template<typename T>
void sparsematrix_copy_host_csr_to_cuda_csr(const SparseMatrix<T>& host_matrix, CudaSparseMatrixCSR<T>& cuda_matrix) {
  assert(host_matrix.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR);

  if (!cuda_matrix.descr) {
    cusparseStatus_t status = cusparseCreateMatDescr(&cuda_matrix.descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("CUSPARSE Matrix descriptor initialization failed");
    }    
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

  cusparseSetMatIndexBase(cuda_matrix.descr, CUSPARSE_INDEX_BASE_ZERO);

  cuda_api_error_check(
    cudaMalloc((void**)&cuda_matrix.row, (host_matrix.rows()+1)*sizeof(int)));
  cuda_api_error_check(
    cudaMalloc((void**)&cuda_matrix.col, (host_matrix.nonZero())*sizeof(int)));
  cuda_api_error_check(
    cudaMalloc((void**)&cuda_matrix.val, (host_matrix.nonZero())*sizeof(T)));

  cuda_api_error_check(cudaMemcpy(cuda_matrix.row, host_matrix.rowPtr(),
        (host_matrix.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

  cuda_api_error_check(cudaMemcpy(cuda_matrix.col, host_matrix.colPtr(),
        (host_matrix.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

  cuda_api_error_check(cudaMemcpy(cuda_matrix.val, host_matrix.valPtr(),
        (host_matrix.nonZero())*sizeof(T), cudaMemcpyHostToDevice));
}

#endif
