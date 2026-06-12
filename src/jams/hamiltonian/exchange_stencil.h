// Copyright 2026 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_STENCIL_H
#define JAMS_HAMILTONIAN_EXCHANGE_STENCIL_H

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <jams/core/hamiltonian.h>
#include <jams/core/interactions.h>

class ExchangeHamiltonian;

class ExchangeStencilHamiltonian : public Hamiltonian {
public:
  ExchangeStencilHamiltonian(const libconfig::Setting& settings, unsigned int size);
  ~ExchangeStencilHamiltonian() override;

  jams::Real calculate_total_energy(jams::Real time, const SpinArray& spins) override;
  void calculate_energies(jams::Real time, const SpinArray& spins) override;
  void calculate_fields(jams::Real time, const SpinArray& spins) override;
  jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;
  jams::Real calculate_energy(int i, jams::Real time) override;
  jams::Real calculate_energy_difference(
      int i,
      const jams::Vec<double, 3>& spin_initial,
      const jams::Vec<double, 3>& spin_final,
      jams::Real time) override;

  EnergyCurrentInteractionSupport energy_current_interaction_support() const override {
    return EnergyCurrentInteractionSupport::Supported;
  }

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

  [[nodiscard]] bool has_sparse_fallback() const {
    return static_cast<bool>(sparse_fallback_);
  }

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

  [[nodiscard]] jams::Vec<jams::Real, 3> calculate_stencil_field_for_site(
      int site,
      const SpinHostView& spins) const;

private:
  struct TemplateKey {
    int basis_site_i = -1;
    int basis_site_j = -1;
    jams::Vec<int, 3> lattice_translation = {0, 0, 0};

    bool operator<(const TemplateKey& other) const;
  };

  void enable_sparse_fallback(const libconfig::Setting& settings, unsigned int size, const std::string& reason);
  [[nodiscard]] bool direct_lattice_layout_is_supported() const;
  void parse_settings(const libconfig::Setting& settings);
  void build_stencil_entries(const std::vector<InteractionData>& interactions);
  [[nodiscard]] bool stencil_entries_match_dense_materials() const;
  void validate_stencil_symmetry() const;
  void validate_no_duplicate_physical_targets() const;
  void validate_stencil_checks(const std::vector<InteractionChecks>& checks) const;
  [[nodiscard]] std::optional<int> stencil_target_site(
      const jams::Vec<int, 3>& source_cell,
      const StencilEntry& entry) const;
  [[nodiscard]] int dense_site_index(int cell_x, int cell_y, int cell_z, int basis_site) const;
  [[nodiscard]] std::size_t num_stencil_entries() const;

  std::unique_ptr<ExchangeHamiltonian> sparse_fallback_;
  std::vector<std::vector<StencilEntry>> stencil_entries_by_basis_;

  jams::Vec<int, 3> lattice_size_ = {0, 0, 0};
  jams::Vec<bool, 3> periodic_boundaries_ = {false, false, false};
  int num_basis_sites_ = 0;

  jams::RealHi interaction_prefactor_ = 1.0;
  jams::RealHi energy_cutoff_ = 0.0;
  jams::RealHi radius_cutoff_ = 100.0;
  jams::RealHi radius_cutoff_tolerance_ = jams::defaults::lattice_tolerance;
  jams::RealHi distance_tolerance_ = jams::defaults::lattice_tolerance;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_STENCIL_H
