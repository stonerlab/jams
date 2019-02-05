#ifndef JAMS_CUDA_SPARSE_HELPERS_H
#define JAMS_CUDA_SPARSE_HELPERS_H

#include <cuda_runtime.h>
#include <cusparse.h>

#include "jams/containers/sparsematrix.h"
#include "cuda_defs.h"

template<typename T>
struct CudaSparseMatrixCSR {
  int     *row = nullptr;
  int     *col = nullptr;
  T       *val = nullptr;
  cusparseMatDescr_t  descr = nullptr;
};

// This function supports mixed precision by converting the host matrix type
template<typename THst, typename TDev>
void sparsematrix_copy_host_csr_to_cuda_csr(const SparseMatrix<THst>& host_matrix, CudaSparseMatrixCSR<TDev>& cuda_matrix) {
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

  cuda_api_error_check(cudaMemcpy(cuda_matrix.row, host_matrix.rowPtr(),
                                  (host_matrix.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

  cuda_api_error_check(cudaMemcpy(cuda_matrix.col, host_matrix.colPtr(),
                                  (host_matrix.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

  cuda_api_error_check(
          cudaMalloc((void**)&cuda_matrix.val, (host_matrix.nonZero())*sizeof(TDev)));

  if (sizeof(THst) == sizeof(TDev)) {
    cuda_api_error_check(cudaMemcpy(cuda_matrix.val, host_matrix.valPtr(),
                                    (host_matrix.nonZero())*sizeof(TDev), cudaMemcpyHostToDevice));
  } else {
    // convert types
    std::vector<TDev> converted_values(host_matrix.nonZero());
    for (auto i = 0; i < host_matrix.nonZero(); ++i) {
      converted_values[i] = host_matrix.val(i);
    }

    cuda_api_error_check(cudaMemcpy(cuda_matrix.val, converted_values.data(),
                                    (converted_values.size())*sizeof(TDev), cudaMemcpyHostToDevice));
  }
}

#endif
