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

ExchangeNeartreeHamiltonian::ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: SparseInteractionHamiltonian(settings, size) {

    std::ofstream debug_file;
    if (debug_is_enabled()) {
      debug_file.open(jams::output::full_path_filename("DEBUG_exchange.dat"));

      std::ofstream pos_file(jams::output::full_path_filename("DEBUG_pos.dat"));
      for (int n = 0; n < globals::lattice->num_materials(); ++n) {
        for (int i = 0; i < globals::num_spins; ++i) {
          if (globals::lattice->lattice_site_material_id(i) == n) {
            pos_file << i << "\t" << globals::lattice->lattice_site_position_cart(i)[0] << "\t" << globals::lattice->lattice_site_position_cart(
                i)[1] << "\t" << globals::lattice->lattice_site_position_cart(
                i)[2] << "\n";
          }
        }
        pos_file << "\n\n";
      }
      pos_file.close();
    }

    energy_cutoff_ = jams::config_optional<double>(settings, "energy_cutoff", 1E-26);
    energy_cutoff_ *= input_energy_unit_conversion_;
    std::cout << "  energy_cutoff " << energy_cutoff_ << "\n";

    shell_width_ = jams::config_optional<double>(settings, "shell_width", 1e-3);
    std::cout << "  shell_width " << shell_width_ << "\n";
    shell_width_ *= input_distance_unit_conversion_;

    // check that no atoms in the unit cell are closer together than the shell width
    for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {
      for (auto j = i+1; j < globals::lattice->num_basis_sites(); ++j) {
        const auto distance = norm(
          globals::lattice->basis_site_atom(i).position_frac
          - globals::lattice->basis_site_atom(j).position_frac);
        if(distance < shell_width_) {
          throw jams::SanityException("Atoms ", i, " and ", j, " in the unit cell are close together (", distance,
                                      ") than the shell_width (", shell_width_, ").\n Check the positions",
                                      "or relax distance_tolerance");
        }
      }
    }

    if (!settings.exists("interactions")) {
      throw jams::ConfigException(settings, "no 'interactions' setting in ExchangeNeartree hamiltonian");
    }

    interaction_list_.resize(globals::lattice->num_materials());

    double max_radius = 0.0;
    for (int i = 0; i < settings["interactions"].getLength(); ++i) {
      std::string type_name_A = settings["interactions"][i][0].c_str();
      std::string type_name_B = settings["interactions"][i][1].c_str();

      if (!globals::lattice->material_exists(type_name_A)) {
        throw std::runtime_error("exchange neartree interaction " +  std::to_string(i) + ": material " + type_name_A + " does not exist in the config");
      }

      if (!globals::lattice->material_exists(type_name_B)) {
        throw std::runtime_error("exchange neartree interaction " +  std::to_string(i) + ": material " + type_name_B + " does not exist in the config");
      }

      double radius = double(settings["interactions"][i][2]) * input_distance_unit_conversion_;

      if (radius > max_radius) {
        max_radius = radius;
      }

      double jij_value = double(settings["interactions"][i][3]) * input_energy_unit_conversion_;

      auto type_id_A = globals::lattice->material_index(type_name_A);
      auto type_id_B = globals::lattice->material_index(type_name_B);

      interaction_list_[type_id_A].emplace_back(InteractionNT{{type_id_A, type_id_B}, radius, jij_value});

      if (type_id_A != type_id_B) {
        interaction_list_[type_id_B].emplace_back(InteractionNT{{type_id_B, type_id_A}, radius, jij_value});
      }
    }

    jams::InteractionNearTree neartree(globals::lattice->get_supercell().a1(),
                                       globals::lattice->get_supercell().a2(),
                                       globals::lattice->get_supercell().a3(),
                                       globals::lattice->periodic_boundaries(),
                                       max_radius + shell_width_,
                                       shell_width_ / 10.0);

    neartree.insert_sites(globals::lattice->lattice_site_positions_cart());

    auto cartesian_positions = globals::lattice->lattice_site_positions_cart();

    int counter = 0;
    std::vector<int> seen_stamp(globals::num_spins, -1);
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto type_i = globals::lattice->lattice_site_material_id(i);

      // for (const auto& interaction : interaction_list_[type_i]) {
      for (const auto& [types, radius, Jij] : interaction_list_[type_i]) {
        assert(types.first != type_i);
        auto neighbours = neartree.shell(cartesian_positions[i], radius, shell_width_);
        for (const auto& [rij, j] : neighbours) {
          if (i == j) {
            continue;
          }

          if (globals::lattice->lattice_site_material_id(j) == types.second) {
            // don't allow self interaction
            if (seen_stamp[j] == i) {
              throw jams::SanityException("multiple interactions between spins ", i, " and ", j);
            }
            seen_stamp[j] = i;

            if ( std::abs(Jij) > energy_cutoff_ ) {
              insert_interaction_tensor(i, j, Jij * kIdentityMat3);
              counter++;
            }

            if (debug_is_enabled()) {
              debug_file << i << "\t" << j << "\t";
              debug_file << globals::lattice->lattice_site_position_cart(i)[0] << "\t";
              debug_file << globals::lattice->lattice_site_position_cart(i)[1] << "\t";
              debug_file << globals::lattice->lattice_site_position_cart(i)[2] << "\t";
              debug_file << globals::lattice->lattice_site_position_cart(j)[0] << "\t";
              debug_file << globals::lattice->lattice_site_position_cart(j)[1] << "\t";
              debug_file << globals::lattice->lattice_site_position_cart(j)[2] << "\n";
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

  std::cout << "  total interactions " << counter << "\n";
  std::cout << "  average interactions per spin " << counter / double(globals::num_spins) << "\n";

  finalize(jams::SparseMatrixSymmetryCheck::Symmetric);
}