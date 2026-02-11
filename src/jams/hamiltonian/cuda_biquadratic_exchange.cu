// biquadratic_exchange.cu                                             -*-C++-*-

#include <jams/hamiltonian/cuda_biquadratic_exchange.h>
#include <jams/core/lattice.h>
#include <jams/core/globals.h>
#include <jams/core/interactions.h>
#include <jams/cuda/cuda_device_vector_ops.h>

#include <fstream>

__global__ void cuda_biquadratic_exchange_field_kernel(
    const unsigned int num_spins, const double * dev_s, const int * dev_rows, const int * dev_cols, const jams::Real * dev_vals, jams::Real * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = 3 * idx;
  if (idx >= num_spins) return;

  jams::Real3 h_i {static_cast<jams::Real>(0.0), static_cast<jams::Real>(0.0), static_cast<jams::Real>(0.0)};
  const jams::Real3 s_i {static_cast<jams::Real>(dev_s[base + 0]), static_cast<jams::Real>(dev_s[base + 1]), static_cast<jams::Real>(dev_s[base + 2])};

  for (auto m = dev_rows[idx]; m < dev_rows[idx + 1]; ++m) {
    auto j = dev_cols[m];
    const jams::Real3 s_j = {static_cast<jams::Real>(dev_s[3*j + 0]), static_cast<jams::Real>(dev_s[3*j + 1]), static_cast<jams::Real>(dev_s[3*j + 2])};
    const jams::Real B_ij = dev_vals[m];
    const jams::Real s_i_dot_s_j = dot(s_i, s_j);

    h_i.x += static_cast<jams::Real>(2.0) * B_ij * s_j.x * s_i_dot_s_j;
    h_i.y += static_cast<jams::Real>(2.0) * B_ij * s_j.y * s_i_dot_s_j;
    h_i.z += static_cast<jams::Real>(2.0) * B_ij * s_j.z * s_i_dot_s_j;
  }

  dev_h[base + 0] = h_i.x;
  dev_h[base + 1] = h_i.y;
  dev_h[base + 2] = h_i.z;
}

CudaBiquadraticExchangeHamiltonian::CudaBiquadraticExchangeHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
: Hamiltonian(settings, size),
  sparse_matrix_builder_(size, size) {
  bool use_symops = true;
  settings.lookupValue("symops", use_symops);

  // this is in the units specified by 'unit_name' in the input
  energy_cutoff_ = 0.0;
  settings.lookupValue("energy_cutoff", energy_cutoff_);
  std::cout << "    interaction energy cutoff " << energy_cutoff_ << "\n";

  radius_cutoff_ = 100.0;  // lattice parameters
  settings.lookupValue("radius_cutoff", radius_cutoff_);
  std::cout << "    interaction radius cutoff " << radius_cutoff_ << "\n";

  distance_tolerance_ = jams::defaults::lattice_tolerance; // fractional coordinate units
  settings.lookupValue("distance_tolerance", distance_tolerance_);
  std::cout << "    distance_tolerance " << distance_tolerance_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);

  // Read in settings for which consistency checks should be performed on
  // interactions. The checks are performed by the interaction functions.
  //
  // The JAMS config settings are:
  //
  // check_no_zero_motif_neighbour_count
  // -----------------------------------
  // If true, an exception will be raised if any motif position has zero
  // neighbours (i.e. it is not included in the interaction list). It may be
  // desirable to zero neighbours, for example if another interaction
  // Hamiltonian is coupling these sites.
  //
  // check_identical_motif_neighbour_count
  // -------------------------------------
  // If true, an exception will be raised if any sites in the lattice which
  // have the same motif position in the unit cell, have different numbers
  // of neighbours.
  // NOTE: This check will only run if periodic boundaries are disabled.
  //
  // check_identical_motif_total_exchange
  // ------------------------------------
  // If true, an exception will be raised in any sites in the lattice which
  // have the same motif position in the unit cell, have different total
  // exchange energy. The total exchange energy is calculated from the absolute
  // sum of the diagonal components of the exchange tensor.
  // NOTE: This check will only run if periodic boundaries are disabled.

  std::vector<InteractionChecks> interaction_checks;

  if (!settings.exists("check_no_zero_motif_neighbour_count")) {
    interaction_checks.push_back(InteractionChecks::kNoZeroMotifNeighbourCount);
  } else {
    if (bool(settings["check_no_zero_motif_neighbour_count"]) == true) {
      interaction_checks.push_back(InteractionChecks::kNoZeroMotifNeighbourCount);
    }
  }

  if (!settings.exists("check_identical_motif_neighbour_count")) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifNeighbourCount);
  } else {
    if (bool(settings["check_identical_motif_neighbour_count"]) == true) {
      interaction_checks.push_back(InteractionChecks::kIdenticalMotifNeighbourCount);
    }
  }

  if (!settings.exists("check_identical_motif_total_exchange")) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifTotalExchange);
  } else {
    if (bool(settings["check_identical_motif_total_exchange"]) == true) {
      interaction_checks.push_back(InteractionChecks::kIdenticalMotifTotalExchange);
    }
  }

  jams::SparseMatrixSymmetryCheck sparse_matrix_checks = jams::SparseMatrixSymmetryCheck::Symmetric;

  if (settings.exists("check_sparse_matrix_symmetry")) {
    if (bool(settings["check_sparse_matrix_symmetry"]) == false) {
      sparse_matrix_checks = jams::SparseMatrixSymmetryCheck::None;
    }
  }

  std::string coordinate_format_name = "CARTESIAN";
  settings.lookupValue("coordinate_format", coordinate_format_name);
  CoordinateFormat coord_format = coordinate_format_from_string(coordinate_format_name);

  std::cout << "    coordinate format: " << to_string(coord_format) << "\n";
  // exc_file is to maintain backwards compatibility
  if (settings.exists("exc_file")) {
    std::cout << "    interaction file name " << settings["exc_file"].c_str() << "\n";
    std::ifstream interaction_file(settings["exc_file"].c_str());
    if (interaction_file.fail()) {
      throw jams::FileException(settings["exc_file"].c_str(), "failed to open interaction file");
    }
    neighbour_list_ = generate_neighbour_list(
        interaction_file, coord_format, use_symops, energy_cutoff_,radius_cutoff_, distance_tolerance_, interaction_checks);
  } else if (settings.exists("interactions")) {
    neighbour_list_ = generate_neighbour_list(
        settings["interactions"], coord_format, use_symops, energy_cutoff_, radius_cutoff_, distance_tolerance_, interaction_checks);
  } else {
    throw std::runtime_error("'exc_file' or 'interactions' settings are required for exchange hamiltonian");
  }

  std::cout << "    computed interactions: "<< neighbour_list_.size() << "\n";
  std::cout << "    neighbour list memory: " << neighbour_list_.memory() / kBytesToMegaBytes << " MB" << std::endl;

  std::cout << "    interactions per motif position: \n";
  if (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2) && !globals::lattice->has_impurities()) {
    for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {
      std::cout << "      " << i << ": " << neighbour_list_.num_interactions(i) <<"\n";
    }
  }

  for (auto n = 0; n < neighbour_list_.size(); ++n) {
    auto i = neighbour_list_[n].first[0];
    auto j = neighbour_list_[n].first[1];
    auto value = input_energy_unit_conversion_ * neighbour_list_[n].second[0][0];
    if (value > energy_cutoff_ * input_energy_unit_conversion_ ) {
      sparse_matrix_builder_.insert(i, j, value);
    }
  }

  switch(sparse_matrix_checks) {
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


void CudaBiquadraticExchangeHamiltonian::calculate_fields(jams::Real time) {
  assert(is_finalized_);

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_biquadratic_exchange_field_kernel<<<grid_size, block_size>>>
      (globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
       interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
       field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

}


jams::Real CudaBiquadraticExchangeHamiltonian::calculate_total_energy(jams::Real time) {
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
    total_energy += -jams::dot(s_i, 0.5*h_i);
  }
  return 0.5 * total_energy;
}


Vec3R CudaBiquadraticExchangeHamiltonian::calculate_field(int i, jams::Real time) {
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
      field[n] += 2.0 * B_ij * s(j,n) * jams::dot(s_i, s_j);
    }
  }

  return jams::array_cast<jams::Real>(field);
}


jams::Real CudaBiquadraticExchangeHamiltonian::calculate_energy(int i, jams::Real time) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  auto field = calculate_field(i, time);
  return -0.5*jams::dot(s_i, field);
}


jams::Real CudaBiquadraticExchangeHamiltonian::calculate_energy_difference(int i,
                                                                       const Vec3 &spin_initial,
                                                                       const Vec3 &spin_final,
                                                                       jams::Real time) {
  assert(is_finalized_);
  auto field = calculate_field(i, time);
  auto e_initial = -jams::dot(spin_initial, 0.5*field);
  auto e_final = -jams::dot(spin_final, 0.5*field);
  return static_cast<jams::Real>(e_final - e_initial);
}



