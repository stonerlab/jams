// biquadratic_exchange.cu                                             -*-C++-*-

#include <jams/hamiltonian/cuda_biquadratic_exchange.h>
#include <jams/hamiltonian/cuda_biquadratic_exchange_kernel.cuh>

#include <jams/core/lattice.h>

#include <fstream>

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

  std::string coordinate_format_name = "CARTESIAN";
  settings.lookupValue("coordinate_format", coordinate_format_name);
  CoordinateFormat coord_format = coordinate_format_from_string(coordinate_format_name);

  std::cout << "    coordinate format: " << to_string(coord_format) << "\n";
  // exc_file is to maintain backwards compatibility
  if (settings.exists("exc_file")) {
    std::cout << "    interaction file name " << settings["exc_file"].c_str() << "\n";
    std::ifstream interaction_file(settings["exc_file"].c_str());
    if (interaction_file.fail()) {
      jams_die("failed to open interaction file");
    }
    neighbour_list_ = generate_neighbour_list(
        interaction_file, coord_format, use_symops, energy_cutoff_,radius_cutoff_);
  } else if (settings.exists("interactions")) {
    neighbour_list_ = generate_neighbour_list(
        settings["interactions"], coord_format, use_symops, energy_cutoff_, radius_cutoff_);
  } else {
    throw std::runtime_error("'exc_file' or 'interactions' settings are required for exchange hamiltonian");
  }

  std::cout << "    computed interactions: "<< neighbour_list_.size() << "\n";
  std::cout << "    neighbour list memory: " << neighbour_list_.memory() / kBytesToMegaBytes << " MB" << std::endl;

  std::cout << "    interactions per motif position: \n";
  if (lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2) && !lattice->has_impurities()) {
    for (auto i = 0; i < lattice->num_motif_atoms(); ++i) {
      std::cout << "      " << i << ": " << neighbour_list_.num_interactions(i) <<"\n";
    }
  }

  for (auto n = 0; n < neighbour_list_.size(); ++n) {
    auto i = neighbour_list_[n].first[0];
    auto j = neighbour_list_[n].first[1];
    auto value = input_unit_conversion_ * neighbour_list_[n].second[0][0];
    if (value > energy_cutoff_ * input_unit_conversion_ ) {
      sparse_matrix_builder_.insert(i, j, value);
    }
  }

  if (!sparse_matrix_builder_.is_symmetric()) {
    throw std::runtime_error("sparse matrix for " + name() + " is not symmetric");
  }

  interaction_matrix_ = sparse_matrix_builder_
      .set_format(jams::SparseMatrixFormat::CSR)
      .build();
  std::cout << "  " << name() << " sparse matrix memory (CSR): " << memory_in_natural_units(interaction_matrix_.memory()) << "\n";
  sparse_matrix_builder_.clear();
  is_finalized_ = true;
}


void CudaBiquadraticExchangeHamiltonian::calculate_fields() {
  assert(is_finalized_);

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_biquadratic_exchange_field_kernel<<<grid_size, block_size>>>
      (globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
       interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
       field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

}


void CudaBiquadraticExchangeHamiltonian::calculate_energies() {
  assert(is_finalized_);
  // TODO: Add GPU support

  #pragma omp parallel for
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i);
  }
}


double CudaBiquadraticExchangeHamiltonian::calculate_total_energy() {
  using namespace globals;
  assert(is_finalized_);

  calculate_fields();
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


Vec3 CudaBiquadraticExchangeHamiltonian::calculate_field(int i) {
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


double CudaBiquadraticExchangeHamiltonian::calculate_energy(int i) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  auto field = calculate_field(i);
  return -0.5*dot(s_i, field);
}


double CudaBiquadraticExchangeHamiltonian::calculate_energy_difference(int i,
                                                                       const Vec3 &spin_initial,
                                                                       const Vec3 &spin_final) {
  assert(is_finalized_);
  auto field = calculate_field(i);
  auto e_initial = -dot(spin_initial, 0.5*field);
  auto e_final = -dot(spin_final, 0.5*field);
  return e_final - e_initial;
}


