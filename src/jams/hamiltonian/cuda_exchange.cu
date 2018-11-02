#include <iosfwd>

#include "jams/core/solver.h"
#include "jams/helpers/cuda_exception.h"
#include "jams/hamiltonian/cuda_exchange.h"

CudaExchangeHamiltonian::CudaExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: ExchangeHamiltonian(settings, size)
{
    dev_energy_ = jblib::CudaArray<double, 1>(energy_);
    dev_field_  = jblib::CudaArray<double, 1>(field_);

    std::cout << "    init cusparse\n";
    cusparseStatus_t status = cusparseCreate(&cusparse_handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      die("cusparse Library initialization failed");
    }
    cusparseSetStream(cusparse_handle_, dev_stream_.get());

    sparsematrix_copy_host_csr_to_cuda_csr(interaction_matrix_, dev_csr_interaction_matrix_);
}

double CudaExchangeHamiltonian::calculate_total_energy() {
  double total_energy = 0.0;
  calculate_fields();
  dev_field_.copy_to_host_array(field_);
  for (auto i = 0; i < globals::num_spins; ++i) {
    total_energy += -(  globals::s(i,0)*field_(i,0)
                        + globals::s(i,1)*field_(i,1)
                        + globals::s(i,2)*field_(i,2) );
  }
  return 0.5*total_energy;
}

void CudaExchangeHamiltonian::calculate_fields() {
  assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);
  double one = 1.0;
  double zero = 0.0;
  const int num_rows = globals::num_spins3;
  const int num_cols = globals::num_spins3;

  cusparseStatus_t stat =
          cusparseDcsrmv(cusparse_handle_,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  num_rows,
                  num_cols,
                  interaction_matrix_.nonZero(),
                  &one,
                  dev_csr_interaction_matrix_.descr,
                  dev_csr_interaction_matrix_.val,
                  dev_csr_interaction_matrix_.row,
                  dev_csr_interaction_matrix_.col,
                  solver->dev_ptr_spin(),
                  &zero,
                  dev_field_.data());

  if (debug_is_enabled()) {
    if (stat != CUSPARSE_STATUS_SUCCESS) {
      throw cuda_api_exception("cusparse failure", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }
}