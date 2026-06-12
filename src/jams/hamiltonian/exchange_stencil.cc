// Copyright 2026 Joseph Barker. All rights reserved.

#include "jams/hamiltonian/exchange_stencil.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/helpers/output.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

namespace {

jams::Mat<jams::Real, 3, 3> transpose_tensor(const jams::Mat<jams::Real, 3, 3>& tensor) {
  return transpose(tensor);
}

bool tensors_match_for_symmetry(
    const jams::Mat<jams::Real, 3, 3>& a,
    const jams::Mat<jams::Real, 3, 3>& b) {
  const auto scale = std::max<jams::Real>(
      static_cast<jams::Real>(1.0),
      std::max(max_abs(a), max_abs(b)));
  return approximately_equal(a, b, static_cast<jams::Real>(1.0e-4) * scale);
}

jams::Mat<jams::Real, 3, 3> add_tensor(
    const jams::Mat<jams::Real, 3, 3>& lhs,
    const jams::Mat<jams::Real, 3, 3>& rhs) {
  jams::Mat<jams::Real, 3, 3> result = lhs;
  for (auto row = 0; row < 3; ++row) {
    for (auto col = 0; col < 3; ++col) {
      result[row][col] += rhs[row][col];
    }
  }
  return result;
}

}  // namespace

bool ExchangeStencilHamiltonian::TemplateKey::operator<(const TemplateKey& other) const {
  return std::tie(
             basis_site_i,
             basis_site_j,
             lattice_translation[0],
             lattice_translation[1],
             lattice_translation[2])
      < std::tie(
             other.basis_site_i,
             other.basis_site_j,
             other.lattice_translation[0],
             other.lattice_translation[1],
             other.lattice_translation[2]);
}

ExchangeStencilHamiltonian::~ExchangeStencilHamiltonian() = default;

ExchangeStencilHamiltonian::ExchangeStencilHamiltonian(
    const libconfig::Setting& settings,
    const unsigned int size)
    : Hamiltonian(settings, size) {
  lattice_size_ = globals::lattice->size();
  periodic_boundaries_ = globals::lattice->periodic_boundaries();
  num_basis_sites_ = globals::lattice->num_basis_sites();

  parse_settings(settings);

  if (!direct_lattice_layout_is_supported()) {
    enable_sparse_fallback(settings, size, "lattice layout is not dense and regular");
    return;
  }

  const auto use_symops = jams::config_optional<bool>(settings, "symops", true);
  const auto coord_format = jams::config_optional<CoordinateFormat>(
      settings, "coordinate_format", CoordinateFormat::CARTESIAN);

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

  std::vector<InteractionData> interaction_templates;
  if (settings.exists("exc_file")) {
    const auto file_path = jams::config_required<std::string>(settings, "exc_file");
    std::cout << "    interaction file name " << file_path << "\n";
    std::ifstream interaction_file(file_path);
    if (interaction_file.fail()) {
      throw jams::FileException(file_path.c_str(), "failed to open file");
    }
    interaction_templates = generate_interaction_data(
        interaction_file,
        coord_format,
        use_symops,
        energy_cutoff_,
        radius_cutoff_,
        distance_tolerance_,
        interaction_checks,
        radius_cutoff_tolerance_);
  } else if (settings.exists("interactions")) {
    interaction_templates = generate_interaction_data(
        settings["interactions"],
        coord_format,
        use_symops,
        energy_cutoff_,
        radius_cutoff_,
        distance_tolerance_,
        interaction_checks,
        radius_cutoff_tolerance_);
  } else {
    throw jams::ConfigException(settings, "'exc_file' or 'interactions' settings are required");
  }

  build_stencil_entries(interaction_templates);

  if (!stencil_entries_match_dense_materials()) {
    enable_sparse_fallback(settings, size, "interaction template materials do not match the dense basis");
    return;
  }

  if (jams::config_optional<bool>(settings, "check_sparse_matrix_symmetry", true)) {
    validate_stencil_symmetry();
  }
  validate_no_duplicate_physical_targets();
  validate_stencil_checks(interaction_checks);

  std::cout << "    stencil entries: " << num_stencil_entries() << "\n";
  std::cout << "    stencil memory: "
            << memory_in_natural_units(num_stencil_entries() * sizeof(StencilEntry)) << "\n";
  std::cout << "    interactions per motif position: \n";
  for (auto basis = 0; basis < num_basis_sites_; ++basis) {
    std::cout << "      " << basis << ": " << stencil_entries_by_basis_[basis].size() << "\n";
  }
}

void ExchangeStencilHamiltonian::parse_settings(const libconfig::Setting& settings) {
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

  interaction_prefactor_ = jams::config_optional<double>(settings, "interaction_prefactor", 1.0);
  std::cout << "    interaction_prefactor " << interaction_prefactor_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);

  const auto coord_format = jams::config_optional<CoordinateFormat>(
      settings, "coordinate_format", CoordinateFormat::CARTESIAN);
  std::cout << "    coordinate format: " << to_string(coord_format) << "\n";
}

void ExchangeStencilHamiltonian::enable_sparse_fallback(
    const libconfig::Setting& settings,
    const unsigned int size,
    const std::string& reason) {
  std::cout << "    falling back to sparse exchange: " << reason << "\n";
  sparse_fallback_ = std::make_unique<ExchangeHamiltonian>(settings, size);
}

bool ExchangeStencilHamiltonian::direct_lattice_layout_is_supported() const {
  if (globals::lattice->has_impurities() || globals::lattice->has_cropping()) {
    return false;
  }

  const auto expected_num_spins = lattice_size_[0] * lattice_size_[1] * lattice_size_[2] * num_basis_sites_;
  if (globals::num_spins != expected_num_spins) {
    return false;
  }

  for (auto cell_x = 0; cell_x < lattice_size_[0]; ++cell_x) {
    for (auto cell_y = 0; cell_y < lattice_size_[1]; ++cell_y) {
      for (auto cell_z = 0; cell_z < lattice_size_[2]; ++cell_z) {
        for (auto basis = 0; basis < num_basis_sites_; ++basis) {
          const auto site = globals::lattice->site_index_by_unit_cell_optional(cell_x, cell_y, cell_z, basis);
          if (!site || *site != dense_site_index(cell_x, cell_y, cell_z, basis)) {
            return false;
          }

          const auto expected_material = globals::lattice->material_name(
              globals::lattice->basis_site_atom(basis).material_index);
          if (globals::lattice->lattice_site_material_name(*site) != expected_material) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

void ExchangeStencilHamiltonian::build_stencil_entries(
    const std::vector<InteractionData>& interactions) {
  stencil_entries_by_basis_.assign(num_basis_sites_, {});
  std::set<TemplateKey> seen_templates;

  for (const auto& interaction : interactions) {
    const auto tensor = interaction_prefactor_ * input_energy_unit_conversion_
        * interaction.interaction_value_tensor;
    if (max_abs(tensor) <= energy_cutoff_ * input_energy_unit_conversion_) {
      continue;
    }

    TemplateKey key{
        interaction.basis_site_i,
        interaction.basis_site_j,
        interaction.lattice_translation_vector};
    if (!seen_templates.insert(key).second) {
      std::ostringstream message;
      message << "multiple exchange-stencil entries for basis pair "
              << interaction.basis_site_i << " -> " << interaction.basis_site_j
              << " and translation " << interaction.lattice_translation_vector;
      throw std::runtime_error(message.str());
    }

    StencilEntry entry;
    entry.basis_site_j = interaction.basis_site_j;
    entry.lattice_translation = interaction.lattice_translation_vector;
    entry.interaction_vector_cart = interaction.interaction_vector_cart;
    entry.tensor = matrix_cast<jams::Real>(tensor);
    entry.type_i = interaction.type_i;
    entry.type_j = interaction.type_j;
    stencil_entries_by_basis_[interaction.basis_site_i].push_back(std::move(entry));
  }
}

bool ExchangeStencilHamiltonian::stencil_entries_match_dense_materials() const {
  for (auto basis_i = 0; basis_i < num_basis_sites_; ++basis_i) {
    const auto type_i = globals::lattice->material_name(
        globals::lattice->basis_site_atom(basis_i).material_index);
    for (const auto& entry : stencil_entries_by_basis_[basis_i]) {
      const auto type_j = globals::lattice->material_name(
          globals::lattice->basis_site_atom(entry.basis_site_j).material_index);
      if (entry.type_i != type_i || entry.type_j != type_j) {
        return false;
      }
    }
  }
  return true;
}

void ExchangeStencilHamiltonian::validate_stencil_symmetry() const {
  std::map<TemplateKey, const StencilEntry*> entries;
  for (auto basis_i = 0; basis_i < num_basis_sites_; ++basis_i) {
    for (const auto& entry : stencil_entries_by_basis_[basis_i]) {
      entries.insert({TemplateKey{basis_i, entry.basis_site_j, entry.lattice_translation}, &entry});
    }
  }

  for (auto basis_i = 0; basis_i < num_basis_sites_; ++basis_i) {
    for (const auto& entry : stencil_entries_by_basis_[basis_i]) {
      const TemplateKey reverse_key{
          entry.basis_site_j,
          basis_i,
          -entry.lattice_translation};
      const auto reverse = entries.find(reverse_key);
      if (reverse == entries.end()) {
        throw std::runtime_error("exchange-stencil template is not structurally symmetric");
      }

      if (!tensors_match_for_symmetry(
              reverse->second->tensor,
              transpose_tensor(entry.tensor))) {
        throw std::runtime_error("exchange-stencil template is not symmetric");
      }
    }
  }
}

void ExchangeStencilHamiltonian::validate_no_duplicate_physical_targets() const {
  for (auto site = 0; site < globals::num_spins; ++site) {
    const auto source_cell = globals::lattice->cell_offset(site);
    const auto basis = static_cast<int>(globals::lattice->lattice_site_basis_index(site));
    std::set<int> target_sites;

    for (const auto& entry : stencil_entries_by_basis_[basis]) {
      const auto target_site = stencil_target_site(source_cell, entry);
      if (!target_site) {
        continue;
      }

      if (!target_sites.insert(*target_site).second) {
        std::ostringstream message;
        message << "multiple exchange-stencil interactions for sites "
                << site << " and " << *target_site;
        throw std::runtime_error(message.str());
      }
    }
  }
}

void ExchangeStencilHamiltonian::validate_stencil_checks(
    const std::vector<InteractionChecks>& checks) const {
  if (checks.empty()) {
    return;
  }

  std::vector<int> interaction_count(globals::num_spins, 0);
  std::vector<jams::Mat<jams::Real, 3, 3>> total_exchange(globals::num_spins, kZeroMat3R);

  for (auto site = 0; site < globals::num_spins; ++site) {
    const auto source_cell = globals::lattice->cell_offset(site);
    const auto basis = static_cast<int>(globals::lattice->lattice_site_basis_index(site));
    for (const auto& entry : stencil_entries_by_basis_[basis]) {
      const auto target_site = stencil_target_site(source_cell, entry);
      if (!target_site) {
        continue;
      }
      ++interaction_count[site];
      total_exchange[site] = add_tensor(total_exchange[site], entry.tensor);
    }
  }

  for (const auto check : checks) {
    switch (check) {
      case InteractionChecks::kNoZeroMotifNeighbourCount:
        for (auto site = 0; site < globals::num_spins; ++site) {
          if (interaction_count[site] == 0) {
            throw std::runtime_error("inconsistent stencil: some sites have no neighbours");
          }
        }
        break;
      case InteractionChecks::kIdenticalMotifNeighbourCount:
        if (periodic_boundaries_[0] && periodic_boundaries_[1] && periodic_boundaries_[2]) {
          std::vector<int> motif_count(num_basis_sites_, 0);
          for (auto basis = 0; basis < num_basis_sites_; ++basis) {
            motif_count[basis] = interaction_count[basis];
          }
          for (auto site = 0; site < globals::num_spins; ++site) {
            const auto basis = static_cast<int>(globals::lattice->lattice_site_basis_index(site));
            if (interaction_count[site] != motif_count[basis]) {
              throw std::runtime_error(
                  "inconsistent stencil: some sites have different numbers of neighbours for the same motif position");
            }
          }
        }
        break;
      case InteractionChecks::kIdenticalMotifTotalExchange:
        if (periodic_boundaries_[0] && periodic_boundaries_[1] && periodic_boundaries_[2]) {
          std::vector<jams::Mat<jams::Real, 3, 3>> motif_total(num_basis_sites_, kZeroMat3R);
          for (auto basis = 0; basis < num_basis_sites_; ++basis) {
            motif_total[basis] = total_exchange[basis];
          }
          for (auto site = 0; site < globals::num_spins; ++site) {
            const auto basis = static_cast<int>(globals::lattice->lattice_site_basis_index(site));
            if (!approximately_equal(
                    diag(total_exchange[site]),
                    diag(motif_total[basis]),
                    static_cast<jams::Real>(1.0e-6))) {
              throw std::runtime_error("inconsistent stencil: J0");
            }
          }
        }
        break;
    }
  }
}

std::optional<int> ExchangeStencilHamiltonian::stencil_target_site(
    const jams::Vec<int, 3>& source_cell,
    const StencilEntry& entry) const {
  auto target_cell = source_cell + entry.lattice_translation;
  if (!globals::lattice->apply_boundary_conditions(target_cell)) {
    return std::nullopt;
  }
  return globals::lattice->site_index_by_unit_cell_optional(
      target_cell[0],
      target_cell[1],
      target_cell[2],
      entry.basis_site_j);
}

int ExchangeStencilHamiltonian::dense_site_index(
    const int cell_x,
    const int cell_y,
    const int cell_z,
    const int basis_site) const {
  return (((cell_x * lattice_size_[1] + cell_y) * lattice_size_[2] + cell_z) * num_basis_sites_)
      + basis_site;
}

std::size_t ExchangeStencilHamiltonian::num_stencil_entries() const {
  std::size_t total = 0;
  for (const auto& entries : stencil_entries_by_basis_) {
    total += entries.size();
  }
  return total;
}

jams::Vec<jams::Real, 3> ExchangeStencilHamiltonian::calculate_stencil_field_for_site(
    const int site,
    const SpinHostView& spins) const {
  const auto source_cell = globals::lattice->cell_offset(site);
  const auto basis = static_cast<int>(globals::lattice->lattice_site_basis_index(site));
  jams::Vec<jams::Real, 3> field = {0, 0, 0};

  for (const auto& entry : stencil_entries_by_basis_[basis]) {
    if (globals::lattice->lattice_site_material_name(site) != entry.type_i) {
      continue;
    }

    const auto target_site = stencil_target_site(source_cell, entry);
    if (!target_site) {
      continue;
    }

    if (globals::lattice->lattice_site_material_name(*target_site) != entry.type_j) {
      continue;
    }

    const jams::Vec<jams::Real, 3> s_j = {
        spins(*target_site, 0),
        spins(*target_site, 1),
        spins(*target_site, 2)};
    const auto h = entry.tensor * s_j;
    for (auto n = 0; n < 3; ++n) {
      field[n] += h[n];
    }
  }

  return field;
}

void ExchangeStencilHamiltonian::calculate_fields(jams::Real time, const SpinArray& spins) {
  if (sparse_fallback_) {
    sparse_fallback_->calculate_fields(time, spins);
    for (auto site = 0; site < globals::num_spins; ++site) {
      for (auto n = 0; n < 3; ++n) {
        field_(site, n) = sparse_fallback_->field(site, n);
      }
    }
    return;
  }

  const auto spin_view = spins.host_view();
#if HAS_OMP
#pragma omp parallel for
#endif
  for (auto site = 0; site < globals::num_spins; ++site) {
    const auto field = calculate_stencil_field_for_site(site, spin_view);
    for (auto n = 0; n < 3; ++n) {
      field_(site, n) = field[n];
    }
  }
}

void ExchangeStencilHamiltonian::calculate_energies(jams::Real time, const SpinArray& spins) {
  if (sparse_fallback_) {
    sparse_fallback_->calculate_energies(time, spins);
    for (auto site = 0; site < globals::num_spins; ++site) {
      energy_(site) = sparse_fallback_->energy(site);
      for (auto n = 0; n < 3; ++n) {
        field_(site, n) = sparse_fallback_->field(site, n);
      }
    }
    return;
  }

  calculate_fields(time, spins);
  const auto spin_view = spins.host_view();
#if HAS_OMP
#pragma omp parallel for
#endif
  for (auto site = 0; site < globals::num_spins; ++site) {
    const jams::Vec<jams::Real, 3> s_i = {
        spin_view(site, 0),
        spin_view(site, 1),
        spin_view(site, 2)};
    const jams::Vec<jams::Real, 3> h_i = {
        field_(site, 0),
        field_(site, 1),
        field_(site, 2)};
    energy_(site) = static_cast<jams::Real>(-0.5) * jams::dot(s_i, h_i);
  }
}

jams::Real ExchangeStencilHamiltonian::calculate_total_energy(
    jams::Real time,
    const SpinArray& spins) {
  if (sparse_fallback_) {
    return sparse_fallback_->calculate_total_energy(time, spins);
  }

  calculate_energies(time, spins);
  jams::Real total_energy = 0;
#if HAS_OMP
#pragma omp parallel for reduction(+ : total_energy)
#endif
  for (auto site = 0; site < globals::num_spins; ++site) {
    total_energy += energy_(site);
  }
  return total_energy;
}

jams::Vec<jams::Real, 3> ExchangeStencilHamiltonian::calculate_field(
    const int site,
    jams::Real time) {
  if (sparse_fallback_) {
    return sparse_fallback_->calculate_field(site, time);
  }

  return calculate_stencil_field_for_site(site, global_spin_array_for_fields().host_view());
}

jams::Real ExchangeStencilHamiltonian::calculate_energy(const int site, jams::Real time) {
  if (sparse_fallback_) {
    return sparse_fallback_->calculate_energy(site, time);
  }

  const auto field = calculate_field(site, time);
  const jams::Vec<jams::Real, 3> s_i = {
      static_cast<jams::Real>(globals::s(site, 0)),
      static_cast<jams::Real>(globals::s(site, 1)),
      static_cast<jams::Real>(globals::s(site, 2))};
  return static_cast<jams::Real>(-0.5) * jams::dot(s_i, field);
}

jams::Real ExchangeStencilHamiltonian::calculate_energy_difference(
    const int site,
    const jams::Vec<double, 3>& spin_initial,
    const jams::Vec<double, 3>& spin_final,
    jams::Real time) {
  if (sparse_fallback_) {
    return sparse_fallback_->calculate_energy_difference(site, spin_initial, spin_final, time);
  }

  const auto field = calculate_field(site, time);
  const auto e_initial = -jams::dot(spin_initial, field);
  const auto e_final = -jams::dot(spin_final, field);
  return e_final - e_initial;
}

void ExchangeStencilHamiltonian::add_energy_current_interactions(
    jams::EnergyCurrentInteractionSink& sink) const {
  if (sparse_fallback_) {
    sparse_fallback_->add_energy_current_interactions(sink);
    return;
  }

  for (auto cell_x = 0; cell_x < lattice_size_[0]; ++cell_x) {
    for (auto cell_y = 0; cell_y < lattice_size_[1]; ++cell_y) {
      for (auto cell_z = 0; cell_z < lattice_size_[2]; ++cell_z) {
        const jams::Vec<int, 3> source_cell = {cell_x, cell_y, cell_z};
        for (auto basis_i = 0; basis_i < num_basis_sites_; ++basis_i) {
          const auto source_site = globals::lattice->site_index_by_unit_cell_optional(
              cell_x, cell_y, cell_z, basis_i);
          if (!source_site) {
            continue;
          }

          for (const auto& entry : stencil_entries_by_basis_[basis_i]) {
            if (globals::lattice->lattice_site_material_name(*source_site) != entry.type_i) {
              continue;
            }
            const auto target_site = stencil_target_site(source_cell, entry);
            if (!target_site) {
              continue;
            }
            if (globals::lattice->lattice_site_material_name(*target_site) != entry.type_j) {
              continue;
            }

            sink.insert(
                *source_site,
                *target_site,
                entry.interaction_vector_cart,
                matrix_cast<double>(entry.tensor));
          }
        }
      }
    }
  }
}
