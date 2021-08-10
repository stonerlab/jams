#include <set>
#include <fstream>

#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/consts.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "exchange_neartree.h"
#include "jams/helpers/error.h"
#include "jams/helpers/output.h"
#include <jams/lattice/interaction_neartree.h>

using namespace std;

ExchangeNeartreeHamiltonian::ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: SparseInteractionHamiltonian(settings, size) {

    std::ofstream debug_file;

    if (debug_is_enabled()) {
      debug_file.open(jams::output::full_path_filename("DEBUG_exchange.dat"));

      std::ofstream pos_file(jams::output::full_path_filename("DEBUG_pos.dat"));
      for (int n = 0; n < lattice->num_materials(); ++n) {
        for (int i = 0; i < globals::num_spins; ++i) {
          if (lattice->atom_material_id(i) == n) {
            pos_file << i << "\t" <<  lattice->atom_position(i)[0] << "\t" <<  lattice->atom_position(i)[1] << "\t" << lattice->atom_position(i)[2] << "\n";
          }
        }
        pos_file << "\n\n";
      }
      pos_file.close();
    }

    energy_cutoff_ = 1E-26;  // Joules
    if (settings.exists("energy_cutoff")) {
        energy_cutoff_ = settings["energy_cutoff"];
    }
    cout << "interaction energy cutoff" << energy_cutoff_ << "\n";

    distance_tolerance_ = jams::defaults::lattice_tolerance; // fractional coordinate units
    if (settings.exists("distance_tolerance")) {
        distance_tolerance_ = settings["distance_tolerance"];
    }

    cout << "distance_tolerance " << distance_tolerance_ << "\n";

    // check that no atoms in the unit cell are closer together than the distance_tolerance_
    for (auto i = 0; i < lattice->num_motif_atoms(); ++i) {
      for (auto j = i+1; j < lattice->num_motif_atoms(); ++j) {
        const auto distance = norm(lattice->motif_atom(i).position - lattice->motif_atom(j).position);
        if(distance < distance_tolerance_) {
          jams_die("Atoms %d and %d in the unit_cell are closer together (%f) than the distance_tolerance (%f).\n"
                   "Check position file or relax distance_tolerance for exchange module",
                   i, j, distance, distance_tolerance_);
        }
      }
    }

    if (!settings.exists("interactions")) {
      jams_die("No interactions defined in ExchangeNeartree hamiltonian");
    }

    interaction_list_.resize(lattice->num_materials());

    double max_radius = 0.0;

    for (int i = 0; i < settings["interactions"].getLength(); ++i) {
      std::string type_name_A = settings["interactions"][i][0].c_str();
      std::string type_name_B = settings["interactions"][i][1].c_str();

      if (!lattice->material_exists(type_name_A)) {
        throw std::runtime_error("exchange neartree interaction " +  std::to_string(i) + ": material " + type_name_A + " does not exist in the config");
      }

      if (!lattice->material_exists(type_name_B)) {
        throw std::runtime_error("exchange neartree interaction " +  std::to_string(i) + ": material " + type_name_A + " does not exist in the config");
      }

      double inner_radius = settings["interactions"][i][2];
      double outer_radius = settings["interactions"][i][3];

      if (outer_radius > max_radius) {
        max_radius = outer_radius;
      }

      double jij_value = double(settings["interactions"][i][4]) * input_unit_conversion_;

      auto type_id_A = lattice->material_id(type_name_A);
      auto type_id_B = lattice->material_id(type_name_B);

      InteractionNT jij = {type_id_A, type_id_B, inner_radius, outer_radius, jij_value};

      interaction_list_[type_id_A].push_back(jij);
    }

    cout << "\ncomputed interactions\n";

    jams::InteractionNearTree neartree(lattice->get_supercell().a(), lattice->get_supercell().b(), lattice->get_supercell().c(), lattice->periodic_boundaries(), max_radius + distance_tolerance_, jams::defaults::lattice_tolerance);
    neartree.insert_sites(lattice->atom_cartesian_positions());

    int counter = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
      std::vector<bool> is_already_interacting(globals::num_spins, false);

      const auto type_i = lattice->atom_material_id(i);

      for (const auto& interaction : interaction_list_[type_i]) {
        const auto type_j = interaction.material[1];

        auto nbr_lower = neartree.neighbours(lattice->atom_position(i), interaction.inner_radius);
        auto nbr_upper = neartree.neighbours(lattice->atom_position(i), interaction.outer_radius + distance_tolerance_);

//        auto compare_func = [](Atom a, Atom b) { return a.id < b.id; };

        std::sort(nbr_lower.begin(), nbr_lower.end());
        std::sort(nbr_upper.begin(), nbr_upper.end());

        std::vector<std::pair<Vec3, int>> nbr;
        std::set_difference(nbr_upper.begin(), nbr_upper.end(), nbr_lower.begin(), nbr_lower.end(), std::inserter(nbr, nbr.begin()));

        for (const std::pair<Vec3, int>& n : nbr) {
          auto j = n.second;
          if (i == j) {
            continue;
          }

          if (lattice->atom_material_id(j) == type_j) {
            // don't allow self interaction
            if (is_already_interacting[j]) {
              jams_die("Multiple interactions between spins %d and %d.\n", i, j);
            }
            is_already_interacting[j] = true;

            Mat3 Jij = interaction.value * kIdentityMat3;

            if ( max_abs(Jij) > energy_cutoff_ ) {
              insert_interaction_tensor(i, j, Jij);
              counter++;
            }

            if (debug_is_enabled()) {
              debug_file << i << "\t" << j << "\t";
              debug_file << lattice->atom_position(i)[0] << "\t";
              debug_file << lattice->atom_position(i)[1] << "\t";
              debug_file << lattice->atom_position(i)[2] << "\t";
              debug_file << lattice->atom_position(j)[0] << "\t";
              debug_file << lattice->atom_position(j)[1] << "\t";
              debug_file << lattice->atom_position(j)[2] << "\n";
            }
          }
        }
      }

      if (debug_is_enabled()) {
        debug_file << "\n\n";
      }
    }

  if (debug_is_enabled()) {
      debug_file.close();
    }

  cout << "  total interactions " << counter << "\n";
  cout << "  average interactions per spin " << counter / double(globals::num_spins) << "\n";

  finalize(jams::SparseMatrixSymmetryCheck::Symmetric);
}