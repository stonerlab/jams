#include <fstream>

#include <jams/common.h>
#include "sparse_interaction.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include "jams/cuda/cuda_array_kernels.h"


SparseInteractionHamiltonian::SparseInteractionHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size),
      interaction_matrix_builder_(
          size,
          jams::interaction_tensor_storage_from_string(
              jams::config_optional<std::string>(settings, "tensor_storage", "auto")),
          jams::config_optional<double>(settings, "tensor_storage_tolerance", 0.0))
{
}

void SparseInteractionHamiltonian::insert_interaction_scalar(const int i, const int j, const jams::Real &value) {
  assert(!is_finalized_);
  if (value == 0.0) {
    return;
  }
  interaction_matrix_builder_.insert(i, j, value);
}

void SparseInteractionHamiltonian::insert_interaction_tensor(const int i, const int j, const jams::Mat<jams::Real, 3, 3> &value) {
  assert(!is_finalized_);
  interaction_matrix_builder_.insert(i, j, value);
}

void SparseInteractionHamiltonian::calculate_fields(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) {
  assert(is_finalized_);
  #if HAS_CUDA
    if (jams::instance().mode() == jams::Mode::GPU) {
      interaction_matrix_.multiply_gpu(spins, field_, cuda_stream_.get());
      return;
    }
  #endif
  interaction_matrix_.multiply(spins, field_);
}

jams::Vec<jams::Real, 3> SparseInteractionHamiltonian::calculate_field(const int i, jams::Real time) {
  assert(is_finalized_);
  jams::Vec<jams::Real, 3> field;

  field = interaction_matrix_.multiply_row(i, globals::s);
  return field;
}

void SparseInteractionHamiltonian::calculate_energies(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) {
  assert(is_finalized_);
  #if HAS_CUDA
  if (jams::instance().mode() == jams::Mode::GPU) {
    interaction_matrix_.multiply_gpu(spins, field_, cuda_stream_.get());
    cuda_array_dot_product(globals::num_spins, jams::Real(-0.5), spins.device_data(), field_.device_data(), energy_.mutable_device_data(), cuda_stream_.get());
    return;
  }
  #endif
  const auto spin_view = spins.host_view();
  #pragma omp parallel for
  for (int i = 0; i < globals::num_spins; ++i) {
    const auto field = interaction_matrix_.multiply_row(i, spins);
    const jams::Vec<jams::Real, 3> s_i = {spin_view(i, 0), spin_view(i, 1), spin_view(i, 2)};
    energy_(i) = -0.5 * jams::dot(s_i, field);
  }
}

jams::Real SparseInteractionHamiltonian::calculate_energy_difference(int i, const jams::Vec<double, 3> &spin_initial,
                                                                 const jams::Vec<double, 3> &spin_final, jams::Real time) {
  assert(is_finalized_);
  auto field = calculate_field(i, time);
  auto e_initial = -jams::dot(spin_initial, field);
  auto e_final = -jams::dot(spin_final, field);
  return e_final - e_initial;
}

jams::Real SparseInteractionHamiltonian::calculate_energy(const int i, jams::Real time) {
  assert(is_finalized_);
  jams::Vec<double, 3> s_i = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
  auto field = calculate_field(i, time);
  return -0.5 * jams::dot(s_i, field);
}

jams::Real SparseInteractionHamiltonian::calculate_total_energy(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) {
  assert(is_finalized_);

#if HAS_CUDA
  if (jams::instance().mode() == jams::Mode::GPU)
  {
    calculate_energies(time, spins);
    return cuda_reduce_array(energy_.device_data(), globals::num_spins, cuda_stream_.get());
  }
#endif


  jams::Real total_energy = 0.0;
  interaction_matrix_.multiply(spins, field_);
  const auto spin_view = spins.host_view();
  #if HAS_OMP
  #pragma omp parallel for default(none) shared(globals::num_spins, spin_view, field_) reduction(+:total_energy)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    jams::Vec<jams::Real, 3> s_i = {spin_view(i, 0), spin_view(i, 1), spin_view(i, 2)};
    jams::Vec<jams::Real, 3> h_i = {field_(i, 0), field_(i, 1), field_(i, 2)};
    total_energy += -0.5 * jams::dot(s_i, h_i);
  }
  return total_energy;
}

void SparseInteractionHamiltonian::add_energy_current_interactions(jams::EnergyCurrentInteractionSink& sink) const {
  assert(is_finalized_);

  const auto* row_data = interaction_matrix_.row_data();
  const auto* col_data = interaction_matrix_.col_data();

  for (auto i = 0; i < interaction_matrix_.num_rows(); ++i) {
    for (auto block = row_data[i]; block < row_data[i + 1]; ++block) {
      const int j = col_data[block];
      const auto r_ji = globals::lattice->displacement(j, i);
      const auto tensor = interaction_matrix_.block_tensor(block);
      sink.insert(i, j, r_ji, {
          static_cast<double>(tensor[0][0]),
          static_cast<double>(tensor[0][1]),
          static_cast<double>(tensor[0][2]),
          static_cast<double>(tensor[1][0]),
          static_cast<double>(tensor[1][1]),
          static_cast<double>(tensor[1][2]),
          static_cast<double>(tensor[2][0]),
          static_cast<double>(tensor[2][1]),
          static_cast<double>(tensor[2][2])
      });
    }
  }
}

void SparseInteractionHamiltonian::finalize(jams::SparseMatrixSymmetryCheck symmetry_check) {
  assert(!is_finalized_);

  if (debug_is_enabled()) {
    std::ofstream os(jams::output::hamiltonian_filename(name(), "DEBUG_spm", "tsv"));
    interaction_matrix_builder_.output(os);
    os.close();
  }

  switch(symmetry_check) {
    case jams::SparseMatrixSymmetryCheck::None:
      break;
    case jams::SparseMatrixSymmetryCheck::Symmetric:
      if (!interaction_matrix_builder_.is_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not symmetric");
      }
      break;
    case jams::SparseMatrixSymmetryCheck::StructurallySymmetric:
      if (!interaction_matrix_builder_.is_structurally_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not structurally symmetric");
      }
      break;
  }

  interaction_matrix_ = interaction_matrix_builder_.build();
  std::cout << "  " << name() << " block sparse matrix storage: "
            << jams::to_string(interaction_matrix_.storage()) << "\n";
  std::cout << "  " << name() << " block sparse matrix blocks: "
            << interaction_matrix_.num_blocks() << "\n";
  std::cout << "  " << name() << " block sparse matrix memory: "
            << memory_in_natural_units(interaction_matrix_.memory()) << "\n";
  is_finalized_ = true;
}
