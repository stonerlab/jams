#include <fstream>

#include <jams/common.h>
#include "sparse_interaction.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"
#include <jams/core/globals.h>

#include "jams/cuda/cuda_array_kernels.h"


SparseInteractionHamiltonian::SparseInteractionHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size),
      sparse_matrix_builder_(3 * size, 3 * size),
      s_float_(size, 3)
{
}

void SparseInteractionHamiltonian::insert_interaction_scalar(const int i, const int j, const jams::Real &value) {
  assert(!is_finalized_);
  if (value == 0.0) {
    return;
  }
  for (auto m = 0; m < 3; ++m) {
    sparse_matrix_builder_.insert(3 * i + m, 3 * j + m, value);
  }
}

void SparseInteractionHamiltonian::insert_interaction_tensor(const int i, const int j, const Mat3R &value) {
  assert(!is_finalized_);
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      if (value[m][n] != 0.0) {
        sparse_matrix_builder_.insert(3 * i + m, 3 * j + n, value[m][n]);
      }
    }
  }
}

void SparseInteractionHamiltonian::calculate_fields(jams::Real time) {
  assert(is_finalized_);
  #if HAS_CUDA
    if (jams::instance().mode() == jams::Mode::GPU) {
#if DO_MIXED_PRECISION
      cuda_array_double_to_float(globals::s.elements(), globals::s.device_data(), s_float_.device_data(), cuda_stream_.get());
      interaction_matrix_.multiply_gpu(s_float_, field_, jams::instance().cusparse_handle(), cuda_stream_.get());
#else
      interaction_matrix_.multiply_gpu(globals::s, field_, jams::instance().cusparse_handle(), cuda_stream_.get());
#endif
      return;
    }
  #endif
  interaction_matrix_.multiply(globals::s, field_);
}

Vec3R SparseInteractionHamiltonian::calculate_field(const int i, jams::Real time) {
  assert(is_finalized_);
  Vec3R field;

  #if HAS_OMP
  #pragma omp parallel for default(none) shared(globals::s, i, field)
  #endif
  for (auto m = 0; m < 3; ++m) {
    field[m] = interaction_matrix_.multiply_row(3*i + m, globals::s);
  }
  return field;
}

void SparseInteractionHamiltonian::calculate_energies(jams::Real time) {
  assert(is_finalized_);
  #if HAS_CUDA
  if (jams::instance().mode() == jams::Mode::GPU) {
    calculate_fields(time);
    cuda_array_dot_product(globals::num_spins, jams::Real(-0.5), globals::s.device_data(), field_.device_data(), energy_.device_data(), cuda_stream_.get());
    return;
  }
  #endif
  #pragma omp parallel for
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

jams::Real SparseInteractionHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                                 const Vec3 &spin_final, jams::Real time) {
  assert(is_finalized_);
  auto field = calculate_field(i, time);
  auto e_initial = -jams::dot(spin_initial, field);
  auto e_final = -jams::dot(spin_final, field);
  return e_final - e_initial;
}

jams::Real SparseInteractionHamiltonian::calculate_energy(const int i, jams::Real time) {
  assert(is_finalized_);
  Vec3 s_i = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
  auto field = calculate_field(i, time);
  return -0.5 * jams::dot(s_i, field);
}

jams::Real SparseInteractionHamiltonian::calculate_total_energy(jams::Real time) {
  assert(is_finalized_);

#if HAS_CUDA
  if (jams::instance().mode() == jams::Mode::GPU)
  {
    calculate_energies(time);
    return cuda_reduce_array(energy_.device_data(), globals::num_spins, cuda_stream_.get());
  }
#endif


  jams::Real total_energy = 0.0;
  calculate_fields(time);
  #if HAS_OMP
  #pragma omp parallel for default(none) shared(globals::num_spins, globals::s, field_) reduction(+:total_energy)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 s_i = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    Vec3 h_i = {field_(i,0), field_(i, 1), field_(i, 2)};
    total_energy += -jams::dot(s_i, h_i);
  }
  return total_energy;
}

void SparseInteractionHamiltonian::finalize(jams::SparseMatrixSymmetryCheck symmetry_check) {
  assert(!is_finalized_);

  if (debug_is_enabled()) {
    std::ofstream os(jams::output::full_path_filename("DEBUG_" + name() + "_spm.tsv"));
    sparse_matrix_builder_.output(os);
    os.close();
  }

  switch(symmetry_check) {
    case jams::SparseMatrixSymmetryCheck::None:
      break;
    case jams::SparseMatrixSymmetryCheck::Symmetric:
      if (!sparse_matrix_builder_.is_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not symmetric");
      }
      break;
    case jams::SparseMatrixSymmetryCheck::StructurallySymmetric:
      if (!sparse_matrix_builder_.is_structurally_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not structurally symmetric");
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
