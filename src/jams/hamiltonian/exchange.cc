#include <fstream>
#include <set>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {
  const auto use_symops = jams::config_optional<bool>(settings, "symops", true);

  // this is in the units specified by 'unit_name' in the input
  energy_cutoff_ = jams::config_optional<double>(settings, "energy_cutoff", 0.0);
  std::cout << "    interaction energy cutoff " << energy_cutoff_ << "\n";

  radius_cutoff_ = jams::config_optional<double>(settings, "radius_cutoff", 100.0);  // lattice parameters
  std::cout << "    interaction radius cutoff " << radius_cutoff_ << "\n";

  // fractional coordinate units
  distance_tolerance_ = jams::config_optional<double>(
      settings, "distance_tolerance", jams::defaults::lattice_tolerance);
  std::cout << "    distance_tolerance " << distance_tolerance_ << "\n";

  interaction_prefactor_ = jams::config_optional<double>(settings, "interaction_prefactor", 1.0);
  std::cout << "    interaction_prefactor " << interaction_prefactor_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);

  if (debug_is_enabled()) {
    std::ofstream pos_file(jams::output::hamiltonian_filename(name() + "_DEBUG_pos", "tsv"));
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
    pos_file.close();
  }

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

  if (jams::config_optional<bool>(settings, "check_no_zero_motif_neighbour_count", true)) {
    interaction_checks.push_back(InteractionChecks::kNoZeroMotifNeighbourCount);
  }

  if (jams::config_optional<bool>(settings, "check_identical_motif_neighbour_count", true)) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifNeighbourCount);
  }

  if (jams::config_optional<bool>(settings, "check_identical_motif_total_exchange", true)) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifTotalExchange);
  }

  const auto sparse_matrix_checks = jams::config_optional<bool>(settings, "check_sparse_matrix_symmetry", true)
      ? jams::SparseMatrixSymmetryCheck::Symmetric
      : jams::SparseMatrixSymmetryCheck::None;

  const auto coord_format = jams::config_optional<CoordinateFormat>(
      settings, "coordinate_format", CoordinateFormat::CARTESIAN);

  std::cout << "    coordinate format: " << to_string(coord_format) << "\n";
  // exc_file is to maintain backwards compatibility
  if (settings.exists("exc_file")) {
    const auto file_path = jams::config_required<std::string>(settings, "exc_file");
    std::cout << "    interaction file name " << file_path << "\n";
    std::ifstream interaction_file(file_path);
    if (interaction_file.fail()) {
      throw jams::FileException(file_path.c_str(), "failed to open file");
    }
    neighbour_list_ = generate_neighbour_list(
        interaction_file,
        coord_format,
        use_symops,
        energy_cutoff_,
        radius_cutoff_,
        distance_tolerance_,
        interaction_checks);
  } else if (settings.exists("interactions")) {
    neighbour_list_ = generate_neighbour_list(
        settings["interactions"],
        coord_format,
        use_symops,
        energy_cutoff_,
        radius_cutoff_,
        distance_tolerance_,
        interaction_checks);
  } else {
    throw jams::ConfigException(settings, "'exc_file' or 'interactions' settings are required");
  }

  if (debug_is_enabled()) {
    std::ofstream debug_file(jams::output::hamiltonian_filename(name() + "_DEBUG_exchange_nbr_list", "tsv"));
    write_neighbour_list(debug_file, neighbour_list_);
    debug_file.close();
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
    auto Jij = interaction_prefactor_ * input_energy_unit_conversion_ * neighbour_list_[n].second;
    if (max_abs(Jij) > energy_cutoff_ * input_energy_unit_conversion_ ) {
      insert_interaction_tensor(i, j, matrix_cast<jams::Real>(Jij));
    }
  }

  finalize(sparse_matrix_checks);
}

const jams::InteractionList<jams::Mat<double, 3, 3>,2> &ExchangeHamiltonian::neighbour_list() const {
  return neighbour_list_;
}
