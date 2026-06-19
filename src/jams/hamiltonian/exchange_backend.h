// Copyright 2026 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_BACKEND_H
#define JAMS_HAMILTONIAN_EXCHANGE_BACKEND_H

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <jams/containers/block_sparse_interaction_matrix.h>
#include <jams/hamiltonian/exchange.h>

class ExchangeBackendImpl {
public:
  using SpinArray = Hamiltonian::SpinArray;
  using SpinHostView = Hamiltonian::SpinHostView;
  using EnergyArray = jams::MultiArray<jams::Real, 1>;
  using FieldArray = jams::MultiArray<jams::Real, 2>;

  virtual ~ExchangeBackendImpl() = default;

  [[nodiscard]] virtual bool uses_device() const {
    return false;
  }

  [[nodiscard]] virtual bool supports_calculate_fields_in_parallel() const = 0;
  virtual void calculate_fields(jams::Real time, const SpinArray& spins, FieldArray& field) = 0;
  virtual void calculate_fields_in_parallel(jams::Real time, const SpinArray& spins, FieldArray& field) = 0;
  virtual void calculate_energies(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) = 0;
  [[nodiscard]] virtual jams::Real calculate_total_energy(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) = 0;
  [[nodiscard]] virtual jams::Vec<jams::Real, 3> calculate_field(
      int site,
      jams::Real time,
      const SpinArray& spins) = 0;
  [[nodiscard]] virtual jams::Real calculate_energy(
      int site,
      jams::Real time,
      const SpinArray& spins) = 0;
  [[nodiscard]] virtual jams::Real calculate_energy_difference(
      int site,
      const jams::Vec<double, 3>& spin_initial,
      const jams::Vec<double, 3>& spin_final,
      jams::Real time,
      const SpinArray& spins) = 0;
  virtual void add_energy_current_interactions(jams::EnergyCurrentInteractionSink& sink) const = 0;
};

class ExchangeStencilBackend : public ExchangeBackendImpl {
public:
  ExchangeStencilBackend(
      const libconfig::Setting& settings,
      std::shared_ptr<const ExchangeInteractionSetup> setup);
  ~ExchangeStencilBackend() override;

  static ExchangeBackendSupport check_support(const ExchangeInteractionSetup& setup);

  [[nodiscard]] bool supports_calculate_fields_in_parallel() const override;
  void calculate_fields(jams::Real time, const SpinArray& spins, FieldArray& field) override;
  void calculate_fields_in_parallel(jams::Real time, const SpinArray& spins, FieldArray& field) override;
  void calculate_energies(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) override;
  [[nodiscard]] jams::Real calculate_total_energy(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) override;
  [[nodiscard]] jams::Vec<jams::Real, 3> calculate_field(
      int site,
      jams::Real time,
      const SpinArray& spins) override;
  [[nodiscard]] jams::Real calculate_energy(
      int site,
      jams::Real time,
      const SpinArray& spins) override;
  [[nodiscard]] jams::Real calculate_energy_difference(
      int site,
      const jams::Vec<double, 3>& spin_initial,
      const jams::Vec<double, 3>& spin_final,
      jams::Real time,
      const SpinArray& spins) override;
  void add_energy_current_interactions(jams::EnergyCurrentInteractionSink& sink) const override;

protected:
  struct StencilEntry {
    int basis_site_j = -1;
    jams::Vec<int, 3> lattice_translation = {0, 0, 0};
    jams::Vec<double, 3> interaction_vector_cart = {0.0, 0.0, 0.0};
    jams::Mat<jams::Real, 3, 3> tensor = kZeroMat3R;
    std::string type_i;
    std::string type_j;
  };

  [[nodiscard]] int num_basis_sites() const {
    return num_basis_sites_;
  }

  [[nodiscard]] const jams::Vec<int, 3>& lattice_size() const {
    return lattice_size_;
  }

  [[nodiscard]] const jams::Vec<bool, 3>& periodic_boundaries() const {
    return periodic_boundaries_;
  }

  [[nodiscard]] const std::vector<std::vector<StencilEntry>>& stencil_entries_by_basis() const {
    return stencil_entries_by_basis_;
  }

private:
  struct TemplateKey {
    int basis_site_i = -1;
    int basis_site_j = -1;
    jams::Vec<int, 3> lattice_translation = {0, 0, 0};

    bool operator<(const TemplateKey& other) const;
  };

  struct RuntimeEntry {
    int basis_site_j = -1;
    std::array<jams::Real, 9> values{};
  };

  void initialize_from_setup(const libconfig::Setting& settings);
  [[nodiscard]] static bool direct_lattice_layout_is_supported(
      const jams::Vec<int, 3>& lattice_size,
      int num_basis_sites);
  [[nodiscard]] static std::vector<std::vector<StencilEntry>> make_stencil_entries(
      const ExchangeInteractionSetup& setup,
      int num_basis_sites);
  [[nodiscard]] static bool stencil_entries_match_dense_materials(
      const std::vector<std::vector<StencilEntry>>& entries_by_basis);
  void build_runtime_entries();
  void validate_stencil_symmetry() const;
  void validate_no_duplicate_physical_targets() const;
  void validate_stencil_checks(const std::vector<InteractionChecks>& checks) const;
  [[nodiscard]] std::optional<int> stencil_target_site(
      const jams::Vec<int, 3>& source_cell,
      const StencilEntry& entry) const;
  [[nodiscard]] std::size_t num_stencil_entries() const;
  [[nodiscard]] jams::Vec<jams::Real, 3> calculate_stencil_field_for_site(
      int site,
      const SpinHostView& spins) const;
  template <jams::InteractionTensorStorage Storage, bool FullyPeriodic>
  void calculate_fields_storage(const SpinHostView& spins, FieldArray& field);
  template <jams::InteractionTensorStorage Storage, bool FullyPeriodic>
  void calculate_fields_storage_in_parallel(const SpinHostView& spins, FieldArray& field);
  template <jams::InteractionTensorStorage Storage, bool FullyPeriodic>
  [[nodiscard]] jams::Vec<jams::Real, 3> calculate_stencil_field_for_site_storage(
      int site,
      const SpinHostView& spins) const;

  std::shared_ptr<const ExchangeInteractionSetup> setup_;
  std::vector<std::vector<StencilEntry>> stencil_entries_by_basis_;
  jams::Vec<int, 3> lattice_size_ = {0, 0, 0};
  jams::Vec<bool, 3> periodic_boundaries_ = {false, false, false};
  int num_basis_sites_ = 0;
  int num_cells_ = 0;
  jams::RealHi interaction_prefactor_ = 1.0;
  jams::RealHi energy_cutoff_ = 0.0;
  bool fully_periodic_ = false;
  jams::InteractionTensorStorage tensor_storage_ = jams::InteractionTensorStorage::Isotropic;
  int components_per_entry_ = 1;
  int num_cell_translations_ = 0;
  std::vector<int> target_cell_by_translation_;
  std::vector<int> runtime_group_offsets_;
  std::vector<int> runtime_group_entry_offsets_;
  std::vector<int> runtime_group_translation_ids_;
  std::vector<int> runtime_group_target_cell_offsets_;
  std::vector<int> runtime_target_basis_;
  std::vector<int> runtime_target_spin_offsets_;
  std::vector<jams::Real> runtime_values_;
};

#if HAS_CUDA
std::unique_ptr<ExchangeBackendImpl> make_cuda_exchange_stencil_backend(
    const libconfig::Setting& settings,
    std::shared_ptr<const ExchangeInteractionSetup> setup,
    cudaStream_t stream);
#endif

#endif  // JAMS_HAMILTONIAN_EXCHANGE_BACKEND_H
