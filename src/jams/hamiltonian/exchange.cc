#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_backend.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/output.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#if HAS_CUDA
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/cuda/cuda_common.h"
#endif

namespace {

std::vector<InteractionChecks> read_exchange_interaction_checks(
    const libconfig::Setting& settings) {
  std::vector<InteractionChecks> interaction_checks;

  if (jams::config_optional<bool>(settings, "check_no_zero_motif_neighbour_count", true)) {
    interaction_checks.push_back(InteractionChecks::kNoZeroMotifNeighbourCount);
  }

  if (jams::config_optional<bool>(settings, "check_identical_motif_neighbour_count", true)) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifNeighbourCount);
  }

  if (jams::config_optional<bool>(settings, "check_identical_motif_total_exchange", true)) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifTotalExchange);
  }

  return interaction_checks;
}

void write_debug_positions_if_enabled(
    const bool debug_enabled,
    const std::string& hamiltonian_name) {
  if (!debug_enabled) {
    return;
  }

  std::ofstream pos_file(jams::output::hamiltonian_filename(hamiltonian_name, "DEBUG_pos", "tsv"));
  for (int n = 0; n < globals::lattice->num_materials(); ++n) {
    for (int i = 0; i < globals::num_spins; ++i) {
      if (globals::lattice->lattice_site_material_id(i) == n) {
        pos_file << i << "\t" << globals::lattice->lattice_site_position_cart(i) << " | "
                 << globals::lattice->cartesian_to_fractional(
                     globals::lattice->lattice_site_position_cart(i)) << "\n";
      }
    }
    pos_file << "\n\n";
  }
}

bool sparse_only_settings_are_present(const libconfig::Setting& settings) {
  return settings.exists("tensor_storage") || settings.exists("tensor_storage_tolerance");
}

constexpr int kExchangeBenchmarkWarmupCalculations = 10;
constexpr int kExchangeBenchmarkTimedCalculations = 100;

}  // namespace

ExchangeInteractionSetup::ExchangeInteractionSetup(
    const libconfig::Setting& settings,
    const double input_energy_unit_conversion,
    const bool debug_enabled,
    const std::string& hamiltonian_name)
    : debug_enabled_(debug_enabled),
      hamiltonian_name_(hamiltonian_name),
      input_energy_unit_conversion_(input_energy_unit_conversion) {
  const auto use_symops = jams::config_optional<bool>(settings, "symops", true);

  energy_cutoff_ = jams::config_optional<double>(settings, "energy_cutoff", 0.0);
  std::cout << "    interaction energy cutoff " << energy_cutoff_ << "\n";

  radius_cutoff_ = jams::config_optional<double>(settings, "radius_cutoff", 100.0);
  std::cout << "    interaction radius cutoff " << radius_cutoff_ << "\n";

  radius_cutoff_tolerance_ = jams::config_optional<double>(
      settings, "radius_cutoff_tolerance", jams::defaults::lattice_tolerance);
  if (radius_cutoff_tolerance_ < 0.0) {
    throw jams::ConfigException(settings, "radius_cutoff_tolerance must be non-negative");
  }
  std::cout << "    interaction radius cutoff tolerance " << radius_cutoff_tolerance_ << "\n";

  distance_tolerance_ = jams::config_optional<double>(
      settings, "distance_tolerance", jams::defaults::lattice_tolerance);
  std::cout << "    distance_tolerance " << distance_tolerance_ << "\n";

  jams::require_mutually_exclusive_settings(settings, {"prefactor", "interaction_prefactor"});
  interaction_prefactor_ = settings.exists("prefactor")
      ? jams::config_required<double>(settings, "prefactor")
      : jams::config_optional<double>(settings, "interaction_prefactor", 1.0);
  std::cout << "    interaction_prefactor " << interaction_prefactor_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);
  write_debug_positions_if_enabled(debug_enabled_, hamiltonian_name_);

  interaction_checks_ = read_exchange_interaction_checks(settings);

  sparse_matrix_symmetry_check_ = jams::config_optional<bool>(
      settings,
      "check_sparse_matrix_symmetry",
      true)
      ? jams::SparseMatrixSymmetryCheck::Symmetric
      : jams::SparseMatrixSymmetryCheck::None;

  const auto coord_format = jams::config_optional<CoordinateFormat>(
      settings, "coordinate_format", CoordinateFormat::CARTESIAN);
  std::cout << "    coordinate format: " << to_string(coord_format) << "\n";

  if (settings.exists("exc_file")) {
    const auto file_path = jams::config_required<std::string>(settings, "exc_file");
    std::cout << "    interaction file name " << file_path << "\n";
    std::ifstream interaction_file(file_path);
    if (interaction_file.fail()) {
      throw jams::FileException(file_path.c_str(), "failed to open file");
    }
    interaction_templates_ = generate_interaction_data(
        interaction_file,
        coord_format,
        use_symops,
        energy_cutoff_,
        radius_cutoff_,
        distance_tolerance_,
        interaction_checks_,
        radius_cutoff_tolerance_);
  } else if (settings.exists("interactions")) {
    interaction_templates_ = generate_interaction_data(
        const_cast<libconfig::Setting&>(settings["interactions"]),
        coord_format,
        use_symops,
        energy_cutoff_,
        radius_cutoff_,
        distance_tolerance_,
        interaction_checks_,
        radius_cutoff_tolerance_);
  } else {
    throw jams::ConfigException(settings, "'exc_file' or 'interactions' settings are required");
  }
}

const jams::InteractionList<jams::Mat<double, 3, 3>, 2>&
ExchangeInteractionSetup::neighbour_list() const {
  if (!neighbour_list_) {
    neighbour_list_ = neighbour_list_from_interactions(interaction_templates_);
    neighbour_list_checks(*neighbour_list_, interaction_checks_);

    if (debug_enabled_) {
      std::ofstream debug_file(
          jams::output::hamiltonian_filename(hamiltonian_name_, "DEBUG_exchange_nbr_list", "tsv"));
      write_neighbour_list(debug_file, *neighbour_list_);
    }

    std::cout << "    computed interactions: " << neighbour_list_->size() << "\n";
    std::cout << "    neighbour list memory: "
              << neighbour_list_->memory() / kBytesToMegaBytes << " MB" << std::endl;

    std::cout << "    interactions per motif position: \n";
    if (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) &&
        globals::lattice->is_periodic(2) && !globals::lattice->has_impurities()) {
      for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {
        std::cout << "      " << i << ": " << neighbour_list_->num_interactions(i) << "\n";
      }
    }
  }

  return *neighbour_list_;
}

namespace {

class ExchangeSparseMatrixBackend : public ExchangeBackendImpl {
public:
  ExchangeSparseMatrixBackend(
      const libconfig::Setting& settings,
      std::shared_ptr<const ExchangeInteractionSetup> setup,
      const bool debug_enabled,
      const std::string& hamiltonian_name
#if HAS_CUDA
      ,
      cudaStream_t stream
#endif
      )
      : setup_(std::move(setup)),
        interaction_matrix_builder_(
            globals::num_spins,
            jams::interaction_tensor_storage_from_string(
                jams::config_optional<std::string>(settings, "tensor_storage", "auto")),
            jams::config_optional<double>(settings, "tensor_storage_tolerance", 0.0))
#if HAS_CUDA
        ,
        stream_(stream)
#endif
  {
    initialize_from_setup(debug_enabled, hamiltonian_name);
  }

  [[nodiscard]] bool uses_device() const override {
#if HAS_CUDA
    return jams::instance().mode() == jams::Mode::GPU;
#else
    return false;
#endif
  }

  [[nodiscard]] bool supports_calculate_fields_in_parallel() const override {
    return is_finalized_ && !uses_device();
  }

  void calculate_fields(jams::Real, const SpinArray& spins, FieldArray& field) override {
    assert(is_finalized_);
#if HAS_CUDA
    if (uses_device()) {
      interaction_matrix_.multiply_gpu(spins, field, stream_);
      return;
    }
#endif
    interaction_matrix_.multiply(spins, field);
  }

  void calculate_fields_in_parallel(jams::Real, const SpinArray& spins, FieldArray& field) override {
    assert(is_finalized_);
    interaction_matrix_.multiply_in_parallel(spins, field);
  }

  void calculate_energies(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) override {
    assert(is_finalized_);
#if HAS_CUDA
    if (uses_device()) {
      interaction_matrix_.multiply_gpu(spins, field, stream_);
      cuda_array_dot_product(
          globals::num_spins,
          jams::Real(-0.5),
          spins.device_data(),
          field.device_data(),
          energy.mutable_device_data(),
          stream_);
      return;
    }
#endif
    calculate_fields(time, spins, field);
    const auto spin_view = spins.host_view();
    const auto field_view = field.host_view();
    auto energy_view = energy.mutable_host_view();
    const auto* spin_values = spin_view.data();
    const auto* field_values = field_view.data();
    auto* energy_values = energy_view.data();
#if HAS_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < globals::num_spins; ++i) {
      const auto offset = 3 * i;
      energy_values[i] = static_cast<jams::Real>(-0.5)
          * (spin_values[offset] * field_values[offset]
             + spin_values[offset + 1] * field_values[offset + 1]
             + spin_values[offset + 2] * field_values[offset + 2]);
    }
  }

  [[nodiscard]] jams::Real calculate_total_energy(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) override {
    assert(is_finalized_);
#if HAS_CUDA
    if (uses_device()) {
      calculate_energies(time, spins, field, energy);
      return cuda_reduce_array(energy.device_data(), globals::num_spins, stream_);
    }
#endif
    calculate_fields(time, spins, field);
    const auto spin_view = spins.host_view();
    const auto field_view = field.host_view();
    const auto* spin_values = spin_view.data();
    const auto* field_values = field_view.data();
    jams::Real total_energy = 0.0;
#if HAS_OMP
#pragma omp parallel for reduction(+:total_energy)
#endif
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto offset = 3 * i;
      total_energy += static_cast<jams::Real>(-0.5)
          * (spin_values[offset] * field_values[offset]
             + spin_values[offset + 1] * field_values[offset + 1]
             + spin_values[offset + 2] * field_values[offset + 2]);
    }
    return total_energy;
  }

  [[nodiscard]] jams::Vec<jams::Real, 3> calculate_field(
      const int i,
      jams::Real,
      const SpinArray& spins) override {
    assert(is_finalized_);
    return interaction_matrix_.multiply_row(i, spins);
  }

  [[nodiscard]] jams::Real calculate_energy(
      const int i,
      jams::Real time,
      const SpinArray& spins) override {
    assert(is_finalized_);
    const jams::Vec<double, 3> s_i = {spins(i, 0), spins(i, 1), spins(i, 2)};
    const auto field = calculate_field(i, time, spins);
    return static_cast<jams::Real>(-0.5) * jams::dot(s_i, field);
  }

  [[nodiscard]] jams::Real calculate_energy_difference(
      const int i,
      const jams::Vec<double, 3>& spin_initial,
      const jams::Vec<double, 3>& spin_final,
      jams::Real time,
      const SpinArray& spins) override {
    assert(is_finalized_);
    const auto field = calculate_field(i, time, spins);
    const auto e_initial = -jams::dot(spin_initial, field);
    const auto e_final = -jams::dot(spin_final, field);
    return e_final - e_initial;
  }

  void add_energy_current_interactions(jams::EnergyCurrentInteractionSink& sink) const override {
  for (int cell_i_x = 0; cell_i_x < globals::lattice->size(0); ++cell_i_x) {
    for (int cell_i_y = 0; cell_i_y < globals::lattice->size(1); ++cell_i_y) {
      for (int cell_i_z = 0; cell_i_z < globals::lattice->size(2); ++cell_i_z) {
        for (const auto& interaction : setup_->interaction_templates()) {
          const auto site_i = globals::lattice->site_index_by_unit_cell_optional(
              cell_i_x, cell_i_y, cell_i_z, interaction.basis_site_i);
          if (!site_i) {
            continue;
          }

          auto cell_j = jams::Vec<int, 3>{cell_i_x, cell_i_y, cell_i_z}
              + interaction.lattice_translation_vector;
          if (!globals::lattice->apply_boundary_conditions(cell_j)) {
            continue;
          }

          const auto site_j = globals::lattice->site_index_by_unit_cell_optional(
              cell_j[0], cell_j[1], cell_j[2], interaction.basis_site_j);
          if (!site_j) {
            continue;
          }

          if (globals::lattice->lattice_site_material_name(*site_i) != interaction.type_i ||
              globals::lattice->lattice_site_material_name(*site_j) != interaction.type_j) {
            continue;
          }

          const auto Jij = setup_->interaction_prefactor()
              * setup_->input_energy_unit_conversion()
              * interaction.interaction_value_tensor;
          if (max_abs(Jij) <= setup_->energy_cutoff() * setup_->input_energy_unit_conversion()) {
            continue;
          }

          sink.insert(*site_i, *site_j, interaction.interaction_vector_cart, Jij);
        }
      }
    }
  }
  }

private:
  void insert_interaction_tensor(const int i, const int j, const jams::Mat<jams::Real, 3, 3>& value) {
    assert(!is_finalized_);
    interaction_matrix_builder_.insert(i, j, value);
  }

  void initialize_from_setup(const bool debug_enabled, const std::string& hamiltonian_name) {
    const auto& neighbour_list = setup_->neighbour_list();
    for (auto n = 0; n < neighbour_list.size(); ++n) {
      auto i = neighbour_list[n].first[0];
      auto j = neighbour_list[n].first[1];
      auto Jij = setup_->interaction_prefactor()
          * setup_->input_energy_unit_conversion()
          * neighbour_list[n].second;
      if (max_abs(Jij) > setup_->energy_cutoff() * setup_->input_energy_unit_conversion()) {
        insert_interaction_tensor(i, j, matrix_cast<jams::Real>(Jij));
      }
    }

    finalize(setup_->sparse_matrix_symmetry_check(), debug_enabled, hamiltonian_name);
  }

  void finalize(
      const jams::SparseMatrixSymmetryCheck symmetry_check,
      const bool debug_enabled,
      const std::string& hamiltonian_name) {
    assert(!is_finalized_);

    if (debug_enabled) {
      std::ofstream os(jams::output::hamiltonian_filename(hamiltonian_name, "DEBUG_spm", "tsv"));
      interaction_matrix_builder_.output(os);
    }

    switch(symmetry_check) {
      case jams::SparseMatrixSymmetryCheck::None:
        break;
      case jams::SparseMatrixSymmetryCheck::Symmetric:
        if (!interaction_matrix_builder_.is_symmetric()) {
          throw std::runtime_error("sparse matrix for " + hamiltonian_name + " is not symmetric");
        }
        break;
      case jams::SparseMatrixSymmetryCheck::StructurallySymmetric:
        if (!interaction_matrix_builder_.is_structurally_symmetric()) {
          throw std::runtime_error(
              "sparse matrix for " + hamiltonian_name + " is not structurally symmetric");
        }
        break;
    }

    interaction_matrix_ = interaction_matrix_builder_.build();
    std::cout << "  " << hamiltonian_name << " block sparse matrix storage: "
              << jams::to_string(interaction_matrix_.storage()) << "\n";
    std::cout << "  " << hamiltonian_name << " block sparse matrix blocks: "
              << interaction_matrix_.num_blocks() << "\n";
    std::cout << "  " << hamiltonian_name << " block sparse matrix memory: "
              << memory_in_natural_units(interaction_matrix_.memory()) << "\n";
    is_finalized_ = true;
  }

  bool is_finalized_ = false;
  std::shared_ptr<const ExchangeInteractionSetup> setup_;
  jams::BlockSparseInteractionMatrix<jams::Real>::Builder interaction_matrix_builder_;
  jams::BlockSparseInteractionMatrix<jams::Real> interaction_matrix_;
#if HAS_CUDA
  cudaStream_t stream_ = nullptr;
#endif
};

}  // namespace

ExchangeHamiltonian::ExchangeHamiltonian(
    const libconfig::Setting& settings,
    const unsigned int size,
    const bool is_cuda_solver)
    : Hamiltonian(settings, size),
      setup_(std::make_shared<ExchangeInteractionSetup>(
          settings,
          input_energy_unit_conversion_,
          debug_is_enabled(),
          name())) {
  initialize_backend(settings, size, is_cuda_solver);
}

ExchangeHamiltonian::ExchangeHamiltonian(
    const libconfig::Setting& settings,
    const unsigned int size)
    : ExchangeHamiltonian(settings, size, false) {
}

ExchangeHamiltonian::~ExchangeHamiltonian() = default;

ExchangeBackendPolicy ExchangeHamiltonian::parse_backend_policy(
    const libconfig::Setting& settings) {
  const auto backend = lowercase(jams::config_optional<std::string>(settings, "backend", "auto"));
  if (backend == "auto") {
    return ExchangeBackendPolicy::Auto;
  }
  if (backend == "stencil") {
    return ExchangeBackendPolicy::Stencil;
  }
  if (backend == "sparse-matrix") {
    return ExchangeBackendPolicy::SparseMatrix;
  }
  if (backend == "benchmark") {
    return ExchangeBackendPolicy::Benchmark;
  }

  throw jams::ConfigException(
      settings,
      "backend must be one of: auto, stencil, sparse-matrix, benchmark");
}

const char* ExchangeHamiltonian::backend_name(const ExchangeBackend backend) {
  switch (backend) {
    case ExchangeBackend::Stencil:
      return "stencil";
    case ExchangeBackend::SparseMatrix:
      return "sparse-matrix";
  }
  return "unknown";
}

std::unique_ptr<ExchangeBackendImpl> ExchangeHamiltonian::make_sparse_backend(
    const libconfig::Setting& settings) {
  return std::make_unique<ExchangeSparseMatrixBackend>(
      settings,
      setup_,
      debug_is_enabled(),
      name()
#if HAS_CUDA
      ,
      cuda_stream_.get()
#endif
  );
}

std::unique_ptr<ExchangeBackendImpl> ExchangeHamiltonian::make_stencil_backend(
    const libconfig::Setting& settings,
    const bool is_cuda_solver) {
#if HAS_CUDA
  if (is_cuda_solver) {
    return make_cuda_exchange_stencil_backend(settings, setup_, cuda_stream_.get());
  }
#else
  (void)is_cuda_solver;
#endif
  return std::make_unique<ExchangeStencilBackend>(settings, setup_);
}

void ExchangeHamiltonian::synchronize_backend_if_device(const ExchangeBackendImpl& backend) {
#if HAS_CUDA
  if (backend.uses_device()) {
    CHECK_CUDA_STATUS(cudaStreamSynchronize(cuda_stream_.get()));
  }
#else
  (void)backend;
#endif
}

std::unique_ptr<ExchangeBackendImpl> ExchangeHamiltonian::benchmark_and_make_backend(
    const libconfig::Setting& settings,
    const bool is_cuda_solver) {
  auto sparse_backend = make_sparse_backend(settings);
  auto stencil_backend = make_stencil_backend(settings, is_cuda_solver);

  const auto& spins = global_spin_array_for_fields();
  ExchangeBackendImpl::FieldArray sparse_field(globals::num_spins, 3);
  ExchangeBackendImpl::FieldArray stencil_field(globals::num_spins, 3);

  const auto benchmark_candidate = [this, &spins](
      ExchangeBackendImpl& candidate,
      ExchangeBackendImpl::FieldArray& scratch_field) {
    for (int n = 0; n < kExchangeBenchmarkWarmupCalculations; ++n) {
      candidate.calculate_fields(0.0, spins, scratch_field);
    }
    synchronize_backend_if_device(candidate);

    const auto start_time = std::chrono::steady_clock::now();
    for (int n = 0; n < kExchangeBenchmarkTimedCalculations; ++n) {
      candidate.calculate_fields(0.0, spins, scratch_field);
    }
    synchronize_backend_if_device(candidate);
    return std::chrono::steady_clock::now() - start_time;
  };

  const auto sparse_duration = benchmark_candidate(*sparse_backend, sparse_field);
  const auto stencil_duration = benchmark_candidate(*stencil_backend, stencil_field);
  const auto sparse_seconds = std::chrono::duration<double>(sparse_duration).count();
  const auto stencil_seconds = std::chrono::duration<double>(stencil_duration).count();

  std::cout << "    exchange backend benchmark warmup field calculations: "
            << kExchangeBenchmarkWarmupCalculations << "\n";
  std::cout << "    exchange backend benchmark timed field calculations: "
            << kExchangeBenchmarkTimedCalculations << "\n";
  std::cout << "    exchange backend benchmark sparse-matrix: "
            << sparse_seconds << " s\n";
  std::cout << "    exchange backend benchmark stencil: "
            << stencil_seconds << " s\n";

  if (stencil_duration <= sparse_duration) {
    active_backend_ = ExchangeBackend::Stencil;
    std::cout << "    exchange backend benchmark selected stencil\n";
    return stencil_backend;
  }

  active_backend_ = ExchangeBackend::SparseMatrix;
  std::cout << "    exchange backend benchmark selected sparse-matrix\n";
  return sparse_backend;
}

void ExchangeHamiltonian::initialize_backend(
    const libconfig::Setting& settings,
    const unsigned int size,
    const bool is_cuda_solver) {
  (void)size;
  const auto policy = parse_backend_policy(settings);

  if (policy == ExchangeBackendPolicy::SparseMatrix) {
    active_backend_ = ExchangeBackend::SparseMatrix;
  } else {
    const auto stencil_support = ExchangeStencilBackend::check_support(*setup_);
    if (policy == ExchangeBackendPolicy::Benchmark && stencil_support.supported) {
      backend_ = benchmark_and_make_backend(settings, is_cuda_solver);
    } else if (stencil_support.supported) {
      active_backend_ = ExchangeBackend::Stencil;
    } else if (policy == ExchangeBackendPolicy::Stencil) {
      throw jams::ConfigException(
          settings,
          "backend = \"stencil\" is unsafe: ",
          stencil_support.reason);
    } else if (policy == ExchangeBackendPolicy::Benchmark) {
      active_backend_ = ExchangeBackend::SparseMatrix;
      std::cout << "    exchange backend benchmark selected sparse-matrix without timing: "
                << stencil_support.reason << "\n";
    } else {
      active_backend_ = ExchangeBackend::SparseMatrix;
      std::cout << "    exchange backend auto selected sparse-matrix: "
                << stencil_support.reason << "\n";
    }
  }

  if (policy != ExchangeBackendPolicy::Benchmark
      && active_backend_ == ExchangeBackend::Stencil
      && sparse_only_settings_are_present(settings)) {
    throw jams::ConfigException(
        settings,
        "tensor_storage and tensor_storage_tolerance are sparse-matrix backend settings; "
        "set backend = \"sparse-matrix\" to use them");
  }

  std::cout << "    exchange backend " << backend_name(active_backend_) << "\n";

  if (backend_) {
    return;
  }

  switch (active_backend_) {
    case ExchangeBackend::SparseMatrix:
      backend_ = make_sparse_backend(settings);
      break;
    case ExchangeBackend::Stencil:
      backend_ = make_stencil_backend(settings, is_cuda_solver);
      break;
  }
}

const jams::InteractionList<jams::Mat<double, 3, 3>, 2>&
ExchangeHamiltonian::neighbour_list() const {
  return setup_->neighbour_list();
}

void ExchangeHamiltonian::calculate_fields(jams::Real time, const SpinArray& spins) {
  backend_->calculate_fields(time, spins, field_);
#if HAS_CUDA
  if (backend_->uses_device()) {
    record_done();
  }
#endif
}

bool ExchangeHamiltonian::supports_calculate_fields_in_parallel() const {
  return backend_->supports_calculate_fields_in_parallel();
}

void ExchangeHamiltonian::calculate_fields_in_parallel(jams::Real time, const SpinArray& spins) {
  backend_->calculate_fields_in_parallel(time, spins, field_);
}

void ExchangeHamiltonian::calculate_energies(jams::Real time, const SpinArray& spins) {
  backend_->calculate_energies(time, spins, field_, energy_);
#if HAS_CUDA
  if (backend_->uses_device()) {
    record_done();
  }
#endif
}

jams::Real ExchangeHamiltonian::calculate_total_energy(
    jams::Real time,
    const SpinArray& spins) {
  const auto total_energy = backend_->calculate_total_energy(time, spins, field_, energy_);
#if HAS_CUDA
  if (backend_->uses_device()) {
    record_done();
  }
#endif
  return total_energy;
}

jams::Vec<jams::Real, 3> ExchangeHamiltonian::calculate_field(
    const int i,
    jams::Real time) {
  return backend_->calculate_field(i, time, global_spin_array_for_fields());
}

jams::Real ExchangeHamiltonian::calculate_energy(const int i, jams::Real time) {
  return backend_->calculate_energy(i, time, global_spin_array_for_fields());
}

jams::Real ExchangeHamiltonian::calculate_energy_difference(
    const int i,
    const jams::Vec<double, 3>& spin_initial,
    const jams::Vec<double, 3>& spin_final,
    jams::Real time) {
  return backend_->calculate_energy_difference(
      i,
      spin_initial,
      spin_final,
      time,
      global_spin_array_for_fields());
}

void ExchangeHamiltonian::add_energy_current_interactions(
    jams::EnergyCurrentInteractionSink& sink) const {
  backend_->add_energy_current_interactions(sink);
}
