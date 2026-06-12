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

int wrap_periodic_axis_fast(int value, const int size) {
  if (value < 0) {
    value += size;
  } else if (value >= size) {
    value -= size;
  }

  if (value < 0 || value >= size) {
    value %= size;
    if (value < 0) {
      value += size;
    }
  }
  return value;
}

int apply_dense_boundary_axis(
    const int value,
    const int size,
    const bool periodic) {
  if (periodic) {
    return wrap_periodic_axis_fast(value, size);
  }
  if (value < 0 || value >= size) {
    return -1;
  }
  return value;
}

template <jams::InteractionTensorStorage Storage>
constexpr int storage_component_count() {
  if constexpr (Storage == jams::InteractionTensorStorage::Isotropic) {
    return 1;
  } else if constexpr (Storage == jams::InteractionTensorStorage::Anisotropic) {
    return 3;
  } else if constexpr (Storage == jams::InteractionTensorStorage::Symmetric) {
    return 6;
  } else if constexpr (Storage == jams::InteractionTensorStorage::Antisymmetric) {
    return 3;
  } else {
    return 9;
  }
}

template <jams::InteractionTensorStorage Storage>
void accumulate_tensor_field(
    const jams::Real* values,
    const jams::Real sx,
    const jams::Real sy,
    const jams::Real sz,
    jams::Real& hx,
    jams::Real& hy,
    jams::Real& hz) {
  if constexpr (Storage == jams::InteractionTensorStorage::Isotropic) {
    const jams::Real j0 = values[0];
    hx += j0 * sx;
    hy += j0 * sy;
    hz += j0 * sz;
  } else if constexpr (Storage == jams::InteractionTensorStorage::Anisotropic) {
    hx += values[0] * sx;
    hy += values[1] * sy;
    hz += values[2] * sz;
  } else if constexpr (Storage == jams::InteractionTensorStorage::Symmetric) {
    hx += values[0] * sx + values[1] * sy + values[2] * sz;
    hy += values[1] * sx + values[3] * sy + values[4] * sz;
    hz += values[2] * sx + values[4] * sy + values[5] * sz;
  } else if constexpr (Storage == jams::InteractionTensorStorage::Antisymmetric) {
    hx += values[0] * sy + values[1] * sz;
    hy += -values[0] * sx + values[2] * sz;
    hz += -values[1] * sx - values[2] * sy;
  } else {
    hx += values[0] * sx + values[1] * sy + values[2] * sz;
    hy += values[3] * sx + values[4] * sy + values[5] * sz;
    hz += values[6] * sx + values[7] * sy + values[8] * sz;
  }
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
  num_cells_ = lattice_size_[0] * lattice_size_[1] * lattice_size_[2];

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
  build_runtime_entries();

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

void ExchangeStencilHamiltonian::build_runtime_entries() {
  fully_periodic_ = periodic_boundaries_[0] && periodic_boundaries_[1] && periodic_boundaries_[2];
  tensor_storage_ = jams::InteractionTensorStorage::Auto;

  for (const auto& entries : stencil_entries_by_basis_) {
    for (const auto& entry : entries) {
      const auto entry_storage = jams::detail::classify_tensor_storage(entry.tensor, 0.0);
      tensor_storage_ = tensor_storage_ == jams::InteractionTensorStorage::Auto
          ? entry_storage
          : jams::detail::combine_tensor_storage(tensor_storage_, entry_storage);
    }
  }
  if (tensor_storage_ == jams::InteractionTensorStorage::Auto) {
    tensor_storage_ = jams::InteractionTensorStorage::Isotropic;
  }
  components_per_entry_ = jams::interaction_tensor_storage_components(tensor_storage_);

  std::vector<jams::Real> packed_components;
  packed_components.reserve(9);

  std::map<std::tuple<int, int, int>, int> translation_ids;
  std::vector<jams::Vec<int, 3>> translations;
  const auto get_translation_id = [&](const jams::Vec<int, 3>& translation) {
    const auto key = std::make_tuple(translation[0], translation[1], translation[2]);
    const auto found = translation_ids.find(key);
    if (found != translation_ids.end()) {
      return found->second;
    }

    const auto id = static_cast<int>(translations.size());
    translation_ids.emplace(key, id);
    translations.push_back(translation);
    return id;
  };

  runtime_group_offsets_.assign(num_basis_sites_ + 1, 0);
  runtime_group_entry_offsets_.clear();
  runtime_group_translation_ids_.clear();
  runtime_group_target_cell_offsets_.clear();
  runtime_target_basis_.clear();
  runtime_target_spin_offsets_.clear();
  runtime_values_.clear();
  runtime_target_basis_.reserve(num_stencil_entries());
  runtime_target_spin_offsets_.reserve(num_stencil_entries());
  runtime_values_.reserve(num_stencil_entries() * components_per_entry_);
  runtime_group_target_cell_offsets_.reserve(num_stencil_entries());

  for (auto basis = 0; basis < num_basis_sites_; ++basis) {
    runtime_group_offsets_[basis] = static_cast<int>(runtime_group_translation_ids_.size());
    std::map<int, std::vector<RuntimeEntry>> grouped_entries;
    for (const auto& entry : stencil_entries_by_basis_[basis]) {
      RuntimeEntry runtime_entry;
      runtime_entry.basis_site_j = entry.basis_site_j;

      packed_components.clear();
      jams::detail::pack_tensor_components(
          packed_components,
          tensor_storage_,
          entry.tensor,
          0.0);
      std::copy(packed_components.begin(), packed_components.end(), runtime_entry.values.begin());
      grouped_entries[get_translation_id(entry.lattice_translation)].push_back(runtime_entry);
    }

    for (auto& [translation_id, entries] : grouped_entries) {
      runtime_group_entry_offsets_.push_back(static_cast<int>(runtime_target_basis_.size()));
      runtime_group_translation_ids_.push_back(translation_id);
      runtime_group_target_cell_offsets_.push_back(translation_id * num_cells_);
      for (const auto& runtime_entry : entries) {
        runtime_target_basis_.push_back(runtime_entry.basis_site_j);
        runtime_target_spin_offsets_.push_back(3 * runtime_entry.basis_site_j);
        runtime_values_.insert(
            runtime_values_.end(),
            runtime_entry.values.begin(),
            runtime_entry.values.begin() + components_per_entry_);
      }
    }
  }
  runtime_group_offsets_[num_basis_sites_] = static_cast<int>(runtime_group_translation_ids_.size());
  runtime_group_entry_offsets_.push_back(static_cast<int>(runtime_target_basis_.size()));

  num_cell_translations_ = static_cast<int>(translations.size());
  target_cell_by_translation_.assign(num_cell_translations_ * num_cells_, -1);
  for (auto translation_id = 0; translation_id < num_cell_translations_; ++translation_id) {
    const auto& translation = translations[translation_id];
    for (auto cell_x = 0; cell_x < lattice_size_[0]; ++cell_x) {
      for (auto cell_y = 0; cell_y < lattice_size_[1]; ++cell_y) {
        for (auto cell_z = 0; cell_z < lattice_size_[2]; ++cell_z) {
          const int source_cell = (cell_x * lattice_size_[1] + cell_y) * lattice_size_[2] + cell_z;
          const int target_x = apply_dense_boundary_axis(
              cell_x + translation[0],
              lattice_size_[0],
              periodic_boundaries_[0]);
          const int target_y = apply_dense_boundary_axis(
              cell_y + translation[1],
              lattice_size_[1],
              periodic_boundaries_[1]);
          const int target_z = apply_dense_boundary_axis(
              cell_z + translation[2],
              lattice_size_[2],
              periodic_boundaries_[2]);
          if (target_x >= 0 && target_y >= 0 && target_z >= 0) {
            target_cell_by_translation_[translation_id * num_cells_ + source_cell] =
                (target_x * lattice_size_[1] + target_y) * lattice_size_[2] + target_z;
          }
        }
      }
    }
  }

  std::cout << "    stencil tensor storage: " << jams::to_string(tensor_storage_) << "\n";
  std::cout << "    stencil cell translations: " << num_cell_translations_ << "\n";
  std::cout << "    stencil field path: mapped\n";
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
  if (fully_periodic_) {
    switch (tensor_storage_) {
      case jams::InteractionTensorStorage::Isotropic:
        return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Isotropic, true>(site, spins);
      case jams::InteractionTensorStorage::Anisotropic:
        return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Anisotropic, true>(site, spins);
      case jams::InteractionTensorStorage::Symmetric:
        return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Symmetric, true>(site, spins);
      case jams::InteractionTensorStorage::Antisymmetric:
        return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Antisymmetric, true>(site, spins);
      case jams::InteractionTensorStorage::General:
        return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::General, true>(site, spins);
      case jams::InteractionTensorStorage::Auto:
        break;
    }
  }

  switch (tensor_storage_) {
    case jams::InteractionTensorStorage::Isotropic:
      return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Isotropic, false>(site, spins);
    case jams::InteractionTensorStorage::Anisotropic:
      return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Anisotropic, false>(site, spins);
    case jams::InteractionTensorStorage::Symmetric:
      return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Symmetric, false>(site, spins);
    case jams::InteractionTensorStorage::Antisymmetric:
      return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::Antisymmetric, false>(site, spins);
    case jams::InteractionTensorStorage::General:
      return calculate_stencil_field_for_site_storage<jams::InteractionTensorStorage::General, false>(site, spins);
    case jams::InteractionTensorStorage::Auto:
      throw std::runtime_error("cannot calculate exchange-stencil field with auto tensor storage");
  }
  throw std::runtime_error("cannot calculate exchange-stencil field with unknown tensor storage");
}

template <jams::InteractionTensorStorage Storage, bool FullyPeriodic>
jams::Vec<jams::Real, 3> ExchangeStencilHamiltonian::calculate_stencil_field_for_site_storage(
    const int site,
    const SpinHostView& spins) const {
  const int basis = site % num_basis_sites_;
  const int source_cell = site / num_basis_sites_;
  const auto* spin_values = spins.data();
  const auto* tensor_values = runtime_values_.data();
  constexpr int kComponents = storage_component_count<Storage>();

  jams::Real hx = 0;
  jams::Real hy = 0;
  jams::Real hz = 0;

  for (auto group = runtime_group_offsets_[basis]; group < runtime_group_offsets_[basis + 1]; ++group) {
    const int target_cell = target_cell_by_translation_[runtime_group_target_cell_offsets_[group] + source_cell];
    if constexpr (!FullyPeriodic) {
      if (target_cell < 0) {
        continue;
      }
    }

    const int target_spin_base = 3 * target_cell * num_basis_sites_;
    for (auto entry = runtime_group_entry_offsets_[group]; entry < runtime_group_entry_offsets_[group + 1]; ++entry) {
      const int target_offset = target_spin_base + runtime_target_spin_offsets_[entry];
      accumulate_tensor_field<Storage>(
          tensor_values + kComponents * entry,
          spin_values[target_offset],
          spin_values[target_offset + 1],
          spin_values[target_offset + 2],
          hx,
          hy,
          hz);
    }
  }

  return {hx, hy, hz};
}

template <jams::InteractionTensorStorage Storage, bool FullyPeriodic>
void ExchangeStencilHamiltonian::calculate_fields_storage(const SpinHostView& spins) {
  const int num_basis = num_basis_sites_;
  const auto* spin_values = spins.data();
  auto field_view = field_.mutable_host_view();
  auto* field_values = field_view.data();
  const auto* tensor_values = runtime_values_.data();
  constexpr int kComponents = storage_component_count<Storage>();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int source_cell = 0; source_cell < num_cells_; ++source_cell) {
    const int site_base = source_cell * num_basis;
    int field_offset = 3 * site_base;
    for (int basis = 0; basis < num_basis; ++basis, field_offset += 3) {
      jams::Real hx = 0;
      jams::Real hy = 0;
      jams::Real hz = 0;

      for (auto group = runtime_group_offsets_[basis]; group < runtime_group_offsets_[basis + 1]; ++group) {
        const int target_cell = target_cell_by_translation_[runtime_group_target_cell_offsets_[group] + source_cell];
        if constexpr (!FullyPeriodic) {
          if (target_cell < 0) {
            continue;
          }
        }

        const int target_spin_base = 3 * target_cell * num_basis;
        for (auto entry = runtime_group_entry_offsets_[group]; entry < runtime_group_entry_offsets_[group + 1]; ++entry) {
          const int target_offset = target_spin_base + runtime_target_spin_offsets_[entry];
          accumulate_tensor_field<Storage>(
              tensor_values + kComponents * entry,
              spin_values[target_offset],
              spin_values[target_offset + 1],
              spin_values[target_offset + 2],
              hx,
              hy,
              hz);
        }
      }

      field_values[field_offset] = hx;
      field_values[field_offset + 1] = hy;
      field_values[field_offset + 2] = hz;
    }
  }
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

  if (fully_periodic_) {
    switch (tensor_storage_) {
      case jams::InteractionTensorStorage::Isotropic:
        calculate_fields_storage<jams::InteractionTensorStorage::Isotropic, true>(spin_view);
        return;
      case jams::InteractionTensorStorage::Anisotropic:
        calculate_fields_storage<jams::InteractionTensorStorage::Anisotropic, true>(spin_view);
        return;
      case jams::InteractionTensorStorage::Symmetric:
        calculate_fields_storage<jams::InteractionTensorStorage::Symmetric, true>(spin_view);
        return;
      case jams::InteractionTensorStorage::Antisymmetric:
        calculate_fields_storage<jams::InteractionTensorStorage::Antisymmetric, true>(spin_view);
        return;
      case jams::InteractionTensorStorage::General:
        calculate_fields_storage<jams::InteractionTensorStorage::General, true>(spin_view);
        return;
      case jams::InteractionTensorStorage::Auto:
        break;
    }
  }

  switch (tensor_storage_) {
    case jams::InteractionTensorStorage::Isotropic:
      calculate_fields_storage<jams::InteractionTensorStorage::Isotropic, false>(spin_view);
      return;
    case jams::InteractionTensorStorage::Anisotropic:
      calculate_fields_storage<jams::InteractionTensorStorage::Anisotropic, false>(spin_view);
      return;
    case jams::InteractionTensorStorage::Symmetric:
      calculate_fields_storage<jams::InteractionTensorStorage::Symmetric, false>(spin_view);
      return;
    case jams::InteractionTensorStorage::Antisymmetric:
      calculate_fields_storage<jams::InteractionTensorStorage::Antisymmetric, false>(spin_view);
      return;
    case jams::InteractionTensorStorage::General:
      calculate_fields_storage<jams::InteractionTensorStorage::General, false>(spin_view);
      return;
    case jams::InteractionTensorStorage::Auto:
      throw std::runtime_error("cannot calculate exchange-stencil fields with auto tensor storage");
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
  const auto field_view = field_.host_view();
  auto energy_view = energy_.mutable_host_view();
  const auto* spin_values = spin_view.data();
  const auto* field_values = field_view.data();
  auto* energy_values = energy_view.data();
#if HAS_OMP
#pragma omp parallel for
#endif
  for (auto site = 0; site < globals::num_spins; ++site) {
    const auto offset = 3 * site;
    energy_values[site] = static_cast<jams::Real>(-0.5)
        * (spin_values[offset] * field_values[offset]
           + spin_values[offset + 1] * field_values[offset + 1]
           + spin_values[offset + 2] * field_values[offset + 2]);
  }
}

jams::Real ExchangeStencilHamiltonian::calculate_total_energy(
    jams::Real time,
    const SpinArray& spins) {
  if (sparse_fallback_) {
    return sparse_fallback_->calculate_total_energy(time, spins);
  }

  calculate_energies(time, spins);
  const auto energy_view = energy_.host_view();
  const auto* energy_values = energy_view.data();
  jams::Real total_energy = 0;
#if HAS_OMP
#pragma omp parallel for reduction(+ : total_energy)
#endif
  for (auto site = 0; site < globals::num_spins; ++site) {
    total_energy += energy_values[site];
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
