#include <fstream>

#include "sparse_interaction.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

SparseInteractionHamiltonian::SparseInteractionHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size),
      sparse_matrix_builder_(3 * size, 3 * size)
    {
}

void SparseInteractionHamiltonian::insert_interaction_scalar(const int i, const int j, const double &value) {
  assert(!is_finalized_);
  if (value == 0.0) {
    return;
  }
  for (auto m = 0; m < 3; ++m) {
    sparse_matrix_builder_.insert(3 * i + m, 3 * j + m, value);
  }
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
    if (jams::instance().mode() == jams::Mode::GPU) {
      interaction_matrix_.multiply_gpu(globals::s, field_, jams::instance().cusparse_handle(), cusparse_stream_.get());
      return;
    }
  #endif
  interaction_matrix_.multiply(globals::s, field_);
}

Vec3 SparseInteractionHamiltonian::calculate_field(const int i) {
  assert(is_finalized_);
  Vec3 field;
  for (auto m = 0; m < 3; ++m) {
    field[m] = interaction_matrix_.multiply_row(3*i + m, globals::s);
  }
  return field;
}

void SparseInteractionHamiltonian::calculate_energies() {
  assert(is_finalized_);
  // TODO: Add GPU support
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i);
  }
}

double SparseInteractionHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                                 const Vec3 &spin_final) {
  assert(is_finalized_);
  auto field = calculate_field(i);
  auto e_initial = -dot(spin_initial, field);
  auto e_final = -dot(spin_final, field);
  return e_final - e_initial;
}

double SparseInteractionHamiltonian::calculate_energy(const int i) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  auto field = calculate_field(i);
  return -dot(s_i, field);
}

double SparseInteractionHamiltonian::calculate_total_energy() {
  assert(is_finalized_);
  // TODO: Add GPU support
  double total_energy = 0.0;
  #if HAS_OMP
  #pragma omp parallel for reduction(+:total_energy)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    total_energy += calculate_energy(i);
  }
  return 0.5 * total_energy;
}

void SparseInteractionHamiltonian::finalize(jams::SparseMatrixSymmetryCheck symmetry_check) {
  assert(!is_finalized_);

  if (debug_is_enabled()) {
    std::ofstream os(jams::output::full_path_filename("DEBUG_" + name_ + "_spm.tsv"));
    sparse_matrix_builder_.output(os);
    os.close();
  }

  switch(symmetry_check) {
    case jams::SparseMatrixSymmetryCheck::None:
      break;
    case jams::SparseMatrixSymmetryCheck::Symmetric:
      if (!sparse_matrix_builder_.is_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name_ + " is not symmetric");
      }
      break;
    case jams::SparseMatrixSymmetryCheck::StructurallySymmetric:
      if (!sparse_matrix_builder_.is_structurally_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name_ + " is not structurally symmetric");
      }
      break;
  }

  interaction_matrix_ = sparse_matrix_builder_
      .set_format(jams::SparseMatrixFormat::CSR)
      .build();
  std::cout << "  " << name() << " sparse matrix memory (CSR): " << memory_in_natural_units(interaction_matrix_.memory()) << "\n";
  sparse_matrix_builder_.clear();
  is_finalized_ = true;
}
