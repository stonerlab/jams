#include <fstream>
#include <set>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/output.h"

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {
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

  interaction_prefactor_ = 1.0;
  settings.lookupValue("interaction_prefactor", interaction_prefactor_);
  std::cout << "    interaction_prefactor " << interaction_prefactor_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);

  if (debug_is_enabled()) {
    std::ofstream pos_file(jams::output::full_path_filename("DEBUG_pos.tsv"));
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
    auto file_path = settings["exc_file"].c_str();
    std::cout << "    interaction file name " << file_path << "\n";
    std::ifstream interaction_file(file_path);
    if (interaction_file.fail()) {
      throw jams::FileException(file_path, "failed to open file");
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
    std::ofstream debug_file(jams::output::full_path_filename("DEBUG_exchange_nbr_list.tsv"));
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
      insert_interaction_tensor(i, j, Jij);
    }
  }

  finalize(sparse_matrix_checks);
}

const jams::InteractionList<Mat3,2> &ExchangeHamiltonian::neighbour_list() const {
  return neighbour_list_;
}