#include <fstream>

#include "sparse_interaction.h"
#include "jams/core/solver.h"

SparseInteractionHamiltonian::SparseInteractionHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size),
      sparse_matrix_builder_(3 * size, 3 * size)
    {

  #if HAS_CUDA
  CHECK_CUSPARSE_STATUS(cusparseCreate(&cusparse_handle_));
  #endif
}

void SparseInteractionHamiltonian::insert_interaction_tensor(const int i, const int j, const Mat3 &value) {
  assert(!is_finalized_);
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      if (value[m][n] != 0.0) {
        sparse_matrix_builder_.insert(3 * i + m, 3 * j + n, value[m][n]);
      }
    }
  }
}

void SparseInteractionHamiltonian::calculate_fields() {
  assert(is_finalized_);
  #if HAS_CUDA
    if (solver->is_cuda_solver()) {
      interaction_matrix_.multiply_gpu(globals::s, field_, cusparse_handle_, cusparse_stream_.get());
      return;
    }
  #endif
  interaction_matrix_.multiply(globals::s, field_);
}

void SparseInteractionHamiltonian::calculate_one_spin_field(const int i, double *local_field) {
  assert(is_finalized_);
  for (auto m = 0; m < 3; ++m) {
    local_field[m] = interaction_matrix_.multiply_row(3*i + m, globals::s);
  }
}

void SparseInteractionHamiltonian::calculate_energies() {
  assert(is_finalized_);
  // TODO: Add GPU support
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_one_spin_energy(i);
  }
}

double SparseInteractionHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial,
                                                                          const Vec3 &spin_final) {
  assert(is_finalized_);
  double local_field[3];
  calculate_one_spin_field(i, local_field);
  auto e_initial = -dot(spin_initial, local_field);
  auto e_final = -dot(spin_final, local_field);
  return e_final - e_initial;
}

double SparseInteractionHamiltonian::calculate_one_spin_energy(const int i) {
  assert(is_finalized_);
  double local_field[3];
  calculate_one_spin_field(i, local_field);
  return -(globals::s(i,0)*local_field[0] + globals::s(i,1)*local_field[1] + globals::s(i,2)*local_field[2]);
}

double SparseInteractionHamiltonian::calculate_total_energy() {
  assert(is_finalized_);
  // TODO: Add GPU support
  double total_energy = 0.0;
  #if HAS_OMP
  #pragma omp parallel for reduction(+:total_energy)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    total_energy += calculate_one_spin_energy(i);
  }
  return 0.5 * total_energy;
}

void SparseInteractionHamiltonian::finalize() {
  assert(!is_finalized_);

  if (debug_is_enabled()) {
    std::ofstream os(seedname + "_" + name_ + "_spm.tsv");
    sparse_matrix_builder_.output(os);
    os.close();
  }

  if (!sparse_matrix_builder_.is_symmetric()) {
    throw std::runtime_error("sparse matrix for " + name_ + " is not symmetric");
  }
  
  interaction_matrix_ = sparse_matrix_builder_
      .set_format(jams::SparseMatrixFormat::CSR)
      .build();
  std::cout << "    exchange sparse matrix memory (CSR): " << interaction_matrix_.memory() / kBytesToMegaBytes << " (MB)\n";
  sparse_matrix_builder_.clear();
  is_finalized_ = true;
}

SparseInteractionHamiltonian::~SparseInteractionHamiltonian() {
  #ifdef HAS_CUDA
  if (cusparse_handle_ != nullptr) {
    cusparseDestroy(cusparse_handle_);
    cusparse_handle_ = nullptr;
  }
  #endif
}
