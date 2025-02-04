// biquadratic_exchange.cu                                             -*-C++-*-

#include <jams/hamiltonian/cuda_biquadratic_exchange.h>
#include <jams/hamiltonian/cuda_biquadratic_exchange_kernel.cuh>

#include <jams/core/lattice.h>
#include <jams/core/globals.h>
#include <jams/core/interactions.h>

#include <fstream>
#include <jams/helpers/output.h>

CudaBiquadraticExchangeHamiltonian::CudaBiquadraticExchangeHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
: NeighbourListInteractionHamiltonian(settings, size) {
  auto sparse_matrix_symmetry_check = read_sparse_matrix_symmetry_check_from_settings(settings, jams::SparseMatrixSymmetryCheck::Symmetric);

  neighbour_list_ = create_neighbour_list_from_settings(settings);

  print_neighbour_list_info(std::cout, neighbour_list_);

  if (debug_is_enabled()) {
    write_neighbour_list(jams::output::full_path_ofstream("DEBUG_biquadratic_exchange_nbr_list.tsv"), neighbour_list_);
  }

  for (auto n = 0; n < neighbour_list_.size(); ++n) {
    auto i = neighbour_list_[n].first[0];
    auto j = neighbour_list_[n].first[1];
    auto value = input_energy_unit_conversion_ * neighbour_list_[n].second[0][0];

    insert_interaction_scalar(i, j, value);
  }

  finalize(sparse_matrix_symmetry_check);
}


void CudaBiquadraticExchangeHamiltonian::calculate_fields(double time) {
  assert(is_finalized_);

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_biquadratic_exchange_field_kernel<<<grid_size, block_size>>>
      (globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
       interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
       field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

}


Vec3 CudaBiquadraticExchangeHamiltonian::calculate_field(int i, double time) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 field;

  const auto begin = interaction_matrix_.row_data()[i];
  const auto end = interaction_matrix_.row_data()[i+1];

  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  for (auto m = begin; m < end; ++m) {
    auto j = interaction_matrix_.col_data()[m];
    double B_ij = interaction_matrix_.val_data()[m];

    Vec3 s_j = {s(j,0), s(j,1), s(j,2)};

    for (auto n = 0; n < 3; ++n) {
      field[n] += 2.0 * B_ij * s(j,n) * dot(s_i, s_j);
    }
  }

  return field;
}




