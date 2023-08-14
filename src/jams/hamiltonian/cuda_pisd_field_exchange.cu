// biquadratic_exchange.cu                                             -*-C++-*-

#include <jams/hamiltonian/cuda_pisd_field_exchange.h>
#include <jams/hamiltonian/cuda_pisd_field_exchange_kernel.cuh>

#include <jams/core/lattice.h>
#include <jams/core/globals.h>
#include <jams/core/interactions.h>
#include <jams/core/solver.h>

#include <jams/helpers/interaction_list_helpers.h>

#include <fstream>


CudaPISDFieldExchangeHamiltonian::CudaPISDFieldExchangeHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
    : GeneralSparseTwoSiteInteractionHamiltonian<double>(settings, size) {

  applied_field_ = jams::config_required<Vec3>(settings, "applied_field");

  neighbour_list_ = jams::InteractionListFromSettings(settings);

  jams::PrintInteractionListProperties(std::cout, neighbour_list_);

  if (debug_is_enabled()) {
    std::ofstream of(jams::output::full_path_filename(name() + "_hamiltonian_nbr.tsv"));
    jams::PrintInteractionList(of, neighbour_list_);
    of.close();
  }

  for (auto n = 0; n < neighbour_list_.size(); ++n) {
    auto i = neighbour_list_[n].first[0];
    auto j = neighbour_list_[n].first[1];
    auto value = neighbour_list_[n].second[0][0];
    insert_interaction(i, j, value);
  }

  jams::SparseMatrixSymmetryCheck sparse_matrix_checks = jams::SparseMatrixSymmetryCheck::Symmetric;

  if (settings.exists("check_sparse_matrix_symmetry")) {
    if (bool(settings["check_sparse_matrix_symmetry"]) == false) {
      sparse_matrix_checks = jams::SparseMatrixSymmetryCheck::None;
    }
  }

  finalize(sparse_matrix_checks);
}


void CudaPISDFieldExchangeHamiltonian::calculate_fields(double time) {
  assert(is_finalized_);

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  double beta = 1.0 / (kBoltzmannIU * globals::solver->thermostat()->temperature());
  cuda_pisd_field_exchange_field_kernel<<<grid_size, block_size>>>
      (globals::num_spins, applied_field_[0], applied_field_[1], applied_field_[2], beta, globals::mus.device_data(), globals::s.device_data(), interaction_matrix().row_device_data(),
       interaction_matrix().col_device_data(), interaction_matrix().val_device_data(),
       field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

}


void CudaPISDFieldExchangeHamiltonian::calculate_energies(double time) {
  assert(is_finalized_);
  // TODO: Add GPU support

  #pragma omp parallel for
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}


double CudaPISDFieldExchangeHamiltonian::calculate_total_energy(double time) {
  using namespace globals;
  assert(is_finalized_);

  calculate_fields(time);
  double total_energy = 0.0;
  #if HAS_OMP
  #pragma omp parallel for default(none) shared(num_spins, s, field_) reduction(+:total_energy)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
    Vec3 h_i = {field_(i,0), field_(i, 1), field_(i, 2)};
    total_energy += -dot(s_i, 0.5*h_i);
  }
  return 0.5 * total_energy;
}


Vec3 CudaPISDFieldExchangeHamiltonian::calculate_field(int i, double time) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 field;

  const auto begin = interaction_matrix().row_data()[i];
  const auto end = interaction_matrix().row_data()[i+1];

  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  for (auto m = begin; m < end; ++m) {
    auto j = interaction_matrix().col_data()[m];
    double B_ij = interaction_matrix().val_data()[m];

    Vec3 s_j = {s(j,0), s(j,1), s(j,2)};

    for (auto n = 0; n < 3; ++n) {
      field[n] += 2.0 * B_ij * s(j,n) * dot(s_i, s_j);
    }
  }

  return field;
}


double CudaPISDFieldExchangeHamiltonian::calculate_energy(int i, double time) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  auto field = calculate_field(i, time);
  return -0.5*dot(s_i, field);
}


double CudaPISDFieldExchangeHamiltonian::calculate_energy_difference(int i,
                                                                       const Vec3 &spin_initial,
                                                                       const Vec3 &spin_final,
                                                                       double time) {
  assert(is_finalized_);
  auto field = calculate_field(i, time);
  auto e_initial = -dot(spin_initial, 0.5*field);
  auto e_final = -dot(spin_final, 0.5*field);
  return e_final - e_initial;
}



