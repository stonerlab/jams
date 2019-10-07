#include <fstream>
#include <set>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"

using namespace std;

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {
  bool use_symops = true;
  settings.lookupValue("symops", use_symops);

  energy_cutoff_ = 1E-26;  // Joules
  settings.lookupValue("energy_cutoff", energy_cutoff_);
  cout << "    interaction energy cutoff " << energy_cutoff_ << "\n";

  radius_cutoff_ = 100.0;  // lattice parameters
  settings.lookupValue("radius_cutoff", radius_cutoff_);
  cout << "    interaction radius cutoff " << radius_cutoff_ << "\n";

  distance_tolerance_ = jams::defaults::lattice_tolerance; // fractional coordinate units
  settings.lookupValue("distance_tolerance", distance_tolerance_);
  cout << "    distance_tolerance " << distance_tolerance_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);

  if (debug_is_enabled()) {
    std::ofstream pos_file("debug_pos.dat");
    for (int n = 0; n < lattice->num_materials(); ++n) {
      for (int i = 0; i < globals::num_spins; ++i) {
        if (lattice->atom_material_id(i) == n) {
          pos_file << i << "\t" << lattice->atom_position(i) << " | "
                   << lattice->cartesian_to_fractional(lattice->atom_position(i)) << "\n";
        }
      }
      pos_file << "\n\n";
    }
    pos_file.close();
  }

  std::string coordinate_format_name = "CARTESIAN";
  settings.lookupValue("coordinate_format", coordinate_format_name);
  CoordinateFormat coord_format = coordinate_format_from_string(coordinate_format_name);

  cout << "    coordinate format: " << to_string(coord_format) << "\n";
  // exc_file is to maintain backwards compatibility
  if (settings.exists("exc_file")) {
    cout << "    interaction file name " << settings["exc_file"].c_str() << "\n";
    std::ifstream interaction_file(settings["exc_file"].c_str());
    if (interaction_file.fail()) {
      jams_die("failed to open interaction file");
    }
    neighbour_list_ = generate_neighbour_list(
        interaction_file, coord_format, use_symops, energy_cutoff_,radius_cutoff_);
  } else if (settings.exists("interactions")) {
    neighbour_list_ = generate_neighbour_list(
        settings["interactions"], coord_format, use_symops, energy_cutoff_, radius_cutoff_);
  }

  if (debug_is_enabled()) {
    std::ofstream debug_file("DEBUG_exchange_nbr_list.tsv");
    write_neighbour_list(debug_file, neighbour_list_);
    debug_file.close();
  }

  cout << "    computed interactions: "<< neighbour_list_.num_interactions() << "\n";

  cout << "    interactions per motif position: \n";
  if (lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2) && !lattice->has_impurities()) {
    for (auto i = 0; i < lattice->num_motif_atoms(); ++i) {
      cout << "      " << i << ": " << neighbour_list_.num_interactions(i) <<"\n";
    }
  }

  for (int i = 0; i < neighbour_list_.size(); ++i) {
    for (auto const &nbr: neighbour_list_[i]) {
      auto j = nbr.first;
      auto Jij = input_unit_conversion_ * nbr.second;
      if (max_abs(Jij) > energy_cutoff_ / kBohrMagneton ) {
        insert_interaction_tensor(i, j, Jij);
      }
    }
  }

  finalize();
}

const InteractionList<Mat3> &ExchangeHamiltonian::neighbour_list() const {
  return neighbour_list_;
}